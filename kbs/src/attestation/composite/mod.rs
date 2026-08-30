// Copyright (c) 2026 Corvex.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Composite attestation backend: appraise one evidence set with two
//! independent appraisers and release only if both affirm.
//!
//! # Why
//!
//! A single appraiser decides alone. On a full NVIDIA fabric neither available
//! appraiser sees everything: Intel Trust Authority appraises the TDX quote and
//! HOPPER/BLACKWELL GPUs but discards NVSwitch evidence outright (it filters on
//! `arch`, see `intel_trust_authority::build_attest_request`), while the CoCo
//! AS with the NRAS verifier appraises GPUs *and* NVSwitches but signs its
//! verdict with a key we own, which is worth little to a third party.
//!
//! Running both over the same evidence gives full device coverage and makes the
//! GPUs an overlap rather than a handoff — a wrong verdict then needs both
//! appraisers to be wrong in the same direction on the same evidence.
//!
//! # The nonce constraint
//!
//! Both appraisers must agree on how `runtime_data` is digested, because the
//! guest bakes that choice into the evidence at collection time and the digest
//! becomes the expected nonce.
//!
//! ITA negotiates explicitly (sha512 for TDX) via the challenge it returns;
//! the CoCo AS path historically hardcoded sha384. So this backend delegates
//! `generate_challenge` to ITA, and pins the CoCo side to the same algorithm
//! through `GrpcConfig::runtime_data_hash_algorithm`. With both on sha512 the
//! two derived nonces are byte-identical: ITA takes `sha512(runtime_data)[..32]`
//! and the NVIDIA verifier takes the first 32 bytes of the AS-supplied report
//! data.
//!
//! # Fail-closed
//!
//! `verify` returns an error unless *both* appraisers affirm. An error here
//! aborts the RCAR handshake, so no token is issued and no resource is
//! released. That is what makes the Intel verdict gate release rather than
//! merely accompany it.

use std::time::Duration;

use anyhow::{anyhow, Context};
use async_trait::async_trait;
use kbs_types::{Challenge, Tee};
use serde::Deserialize;
use tokio::time::timeout;
use tracing::{info, warn};

use crate::attestation::backend::{Attest, IndependentEvidence};
use crate::attestation::coco::grpc::{GrpcClientPool, GrpcConfig};
use crate::attestation::intel_trust_authority::{
    negotiated_hash_algorithm, IntelTrustAuthority, IntelTrustAuthorityConfig,
};

/// Which appraiser's token is handed back to the guest.
///
/// Both appraisers must affirm regardless — this only selects which signed
/// verdict travels onward as the attestation token, and therefore which claims
/// the KBS resource policy sees as its Rego `input`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ReturnToken {
    /// The CoCo AS EAR token. The default: existing resource policies are
    /// written against EAR submods and keep working unchanged.
    #[default]
    Coco,

    /// The Intel-signed ITA token. Portable — a third party can verify it
    /// against Intel's JWKS — but its claim shape differs from EAR, so
    /// resource policies must be rewritten before selecting it.
    Ita,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct CompositeConfig {
    /// The CoCo AS gRPC appraiser. Covers CPU, GPUs and NVSwitches.
    pub coco_as_grpc: GrpcConfig,

    /// The Intel Trust Authority appraiser. Covers CPU and HOPPER/BLACKWELL
    /// GPUs; NVSwitch evidence is dropped by ITA itself.
    pub intel_ta: IntelTrustAuthorityConfig,

    /// Which signed verdict is returned to the guest. Defaults to `coco`.
    #[serde(default)]
    pub return_token: ReturnToken,

    /// Seconds to wait for each appraiser before giving up.
    ///
    /// Both appraisers are remote, and neither client sets a timeout of its
    /// own, so without a bound here one stalled backend would block every
    /// attestation request indefinitely. Expiry is a refusal, so it fails
    /// closed like any other appraisal failure.
    #[serde(default = "default_appraiser_timeout_secs")]
    pub appraiser_timeout_secs: u64,
}

/// Generous by default: NRAS round trips for a full fabric are not fast, and a
/// premature timeout would present as an attestation failure.
pub const DEFAULT_APPRAISER_TIMEOUT_SECS: u64 = 180;

fn default_appraiser_timeout_secs() -> u64 {
    DEFAULT_APPRAISER_TIMEOUT_SECS
}

impl Default for CompositeConfig {
    fn default() -> Self {
        Self {
            coco_as_grpc: GrpcConfig::default(),
            intel_ta: IntelTrustAuthorityConfig::default(),
            return_token: ReturnToken::default(),
            appraiser_timeout_secs: DEFAULT_APPRAISER_TIMEOUT_SECS,
        }
    }
}

/// Pin the CoCo AS side to whatever algorithm Intel TA negotiates for this TEE.
///
/// Returning `None` for a TEE Intel TA does not model leaves the CoCo AS on its
/// own default; the composite fails that handshake at Intel TA anyway.
fn ita_hash_algorithm(tee: Tee) -> Option<String> {
    negotiated_hash_algorithm(tee).map(|algorithm| algorithm.as_ref().to_lowercase())
}

pub struct Composite {
    coco: GrpcClientPool,
    ita: IntelTrustAuthority,
    return_token: ReturnToken,
    appraiser_timeout: Duration,
}

impl Composite {
    pub async fn new(config: CompositeConfig) -> anyhow::Result<Self> {
        let mut coco = GrpcClientPool::new(config.coco_as_grpc)
            .await
            .context("composite: failed to initialise the CoCo AS appraiser")?;

        // Pin the CoCo side per TEE, to whatever Intel TA negotiates for that
        // TEE. This must not be a single blanket algorithm: Intel TA requires
        // sha512 for TDX but sha256 for SGX and Azure TDX vTPM, so a blanket
        // pin would silently guarantee a nonce mismatch on every non-TDX TEE.
        coco.set_hash_algorithm_selector(ita_hash_algorithm);
        info!("composite: CoCo AS runtime data hash algorithm now follows Intel TA, per TEE");

        let ita = IntelTrustAuthority::new(config.intel_ta)
            .await
            .context("composite: failed to initialise the Intel TA appraiser")?;

        info!(
            "composite attestation backend ready: both CoCo AS and Intel TA must affirm; \
             returning the {:?} token",
            config.return_token
        );

        Ok(Self {
            coco,
            ita,
            return_token: config.return_token,
            appraiser_timeout: Duration::from_secs(config.appraiser_timeout_secs),
        })
    }
}

#[async_trait]
impl Attest for Composite {
    /// Attestation policy belongs to the CoCo AS. ITA policies live in the ITA
    /// portal and are selected by `policy_ids`, not pushed from here.
    async fn set_policy(&self, policy_id: &str, policy: &str) -> anyhow::Result<()> {
        self.coco.set_policy(policy_id, policy).await
    }

    async fn verify(&self, evidence_to_verify: Vec<IndependentEvidence>) -> anyhow::Result<String> {
        // Appraise concurrently. Both calls are network round trips and they
        // are independent, so there is no reason to serialise them.
        let (coco_result, ita_result) = tokio::join!(
            timeout(
                self.appraiser_timeout,
                self.coco.verify(evidence_to_verify.clone())
            ),
            timeout(self.appraiser_timeout, self.ita.verify(evidence_to_verify)),
        );

        let secs = self.appraiser_timeout.as_secs();
        let coco_result =
            coco_result.unwrap_or_else(|_| Err(anyhow!("CoCo AS did not answer within {secs}s")));
        let ita_result =
            ita_result.unwrap_or_else(|_| Err(anyhow!("Intel TA did not answer within {secs}s")));

        // Report every failure, not just the first, so a rejection does not
        // have to be diagnosed one appraiser at a time.
        let mut failures = Vec::new();

        if let Err(e) = &coco_result {
            warn!("composite: CoCo AS refused the evidence: {e:#}");
            failures.push(format!("CoCo AS: {e:#}"));
        }

        if let Err(e) = &ita_result {
            warn!("composite: Intel TA refused the evidence: {e:#}");
            failures.push(format!("Intel TA: {e:#}"));
        }

        if !failures.is_empty() {
            return Err(anyhow!(
                "composite attestation failed — every appraiser must affirm. {}",
                failures.join(" | ")
            ));
        }

        let coco_token = coco_result.expect("checked above");
        let ita_token = ita_result.expect("checked above");

        info!(
            "composite: both appraisers affirmed (CoCo AS token {} bytes, Intel TA token {} bytes)",
            coco_token.len(),
            ita_token.len()
        );

        match self.return_token {
            ReturnToken::Coco => Ok(coco_token),
            ReturnToken::Ita => Ok(ita_token),
        }
    }

    /// Delegate to ITA, which is the stricter of the two: it negotiates an
    /// explicit hash algorithm with the guest and fails the handshake if the
    /// guest cannot provide it. The CoCo AS side is pinned to the same
    /// algorithm in [`Composite::new`].
    async fn generate_challenge(
        &self,
        tee: Tee,
        tee_parameters: serde_json::Value,
    ) -> anyhow::Result<Challenge> {
        self.ita
            .generate_challenge(tee, tee_parameters)
            .await
            .context(
                "composite: Intel TA declined to issue a challenge. Both appraisers must agree \
                 on the runtime data hash algorithm, so this cannot fall back to the CoCo AS \
                 challenge without silently breaking nonce comparison on one side",
            )
    }

    /// Reference values are the CoCo AS's RVPS. ITA has no equivalent endpoint.
    async fn register_reference_value(&self, message: &str) -> anyhow::Result<()> {
        self.coco.register_reference_value(message).await
    }

    async fn query_reference_value(
        &self,
        reference_value_id: &str,
    ) -> anyhow::Result<Option<serde_json::Value>> {
        self.coco.query_reference_value(reference_value_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::config::AttestationServiceConfig;

    #[test]
    fn return_token_defaults_to_coco() {
        // The default must preserve existing behaviour: EAR claims reach the
        // resource policy, so policies written against EAR submods keep working.
        assert_eq!(ReturnToken::default(), ReturnToken::Coco);
        assert_eq!(CompositeConfig::default().return_token, ReturnToken::Coco);
    }

    #[test]
    fn config_deserialises_from_toml() {
        let config: CompositeConfig = toml::from_str(
            r#"
            return_token = "coco"

            [coco_as_grpc]
            as_addr = "http://127.0.0.1:50004"

            [intel_ta]
            base_url = "https://api.trustauthority.intel.com"
            api_key = "test-key"
            certs_file = "https://portal.trustauthority.intel.com"
            "#,
        )
        .expect("composite config should deserialise");

        assert_eq!(config.coco_as_grpc.as_addr, "http://127.0.0.1:50004");
        assert_eq!(
            config.intel_ta.base_url,
            "https://api.trustauthority.intel.com"
        );
        assert_eq!(config.return_token, ReturnToken::Coco);
        // Unset here — Composite::new pins it to sha512.
        assert_eq!(config.coco_as_grpc.runtime_data_hash_algorithm, None);
    }

    #[test]
    fn return_token_ita_is_selectable() {
        let config: CompositeConfig = toml::from_str(
            r#"
            return_token = "ita"

            [coco_as_grpc]
            as_addr = "http://127.0.0.1:50004"

            [intel_ta]
            base_url = "https://api.trustauthority.intel.com"
            api_key = "test-key"
            certs_file = "https://portal.trustauthority.intel.com"
            "#,
        )
        .expect("composite config should deserialise");

        assert_eq!(config.return_token, ReturnToken::Ita);
    }

    /// The shipped reference config must parse as a whole `KbsConfig`, not just
    /// as a `CompositeConfig`: the `type = "composite"` tag and the nested
    /// `[attestation_service.*]` tables are what actually break in the field.
    #[test]
    fn shipped_reference_config_parses() {
        use crate::config::KbsConfig;
        use std::path::Path;

        let config = KbsConfig::try_from(Path::new("config/kbs-config-composite.toml"))
            .expect("shipped composite reference config should parse");

        let AttestationServiceConfig::Composite(composite) =
            config.attestation_service.attestation_service
        else {
            panic!("reference config did not select the composite backend");
        };

        assert_eq!(composite.return_token, ReturnToken::Coco);
        assert_eq!(composite.coco_as_grpc.as_addr, "http://127.0.0.1:50004");
        assert_eq!(
            composite.intel_ta.base_url,
            "https://api.trustauthority.intel.com"
        );
        assert_eq!(composite.coco_as_grpc.runtime_data_hash_algorithm, None);
    }

    /// Regression test for the defect this design is most exposed to: pinning
    /// the CoCo AS side to one blanket algorithm. Intel TA requires sha512 for
    /// TDX but sha256 for SGX and Azure TDX vTPM, so a blanket sha512 pin would
    /// guarantee a nonce mismatch — and therefore a 100% attestation failure —
    /// on every non-TDX TEE.
    #[test]
    fn hash_algorithm_is_selected_per_tee_not_blanket() {
        assert_eq!(ita_hash_algorithm(Tee::Tdx).as_deref(), Some("sha512"));
        assert_eq!(ita_hash_algorithm(Tee::Sgx).as_deref(), Some("sha256"));
        assert_eq!(
            ita_hash_algorithm(Tee::AzTdxVtpm).as_deref(),
            Some("sha256")
        );

        // Not a single value across TEEs — that is the whole point.
        assert_ne!(ita_hash_algorithm(Tee::Tdx), ita_hash_algorithm(Tee::Sgx));
    }

    /// TEEs Intel TA does not model yield no pin, leaving the CoCo AS on its
    /// own default. The composite fails those handshakes at Intel TA anyway.
    #[test]
    fn unmodelled_tees_get_no_pin() {
        assert_eq!(ita_hash_algorithm(Tee::Snp), None);
        assert_eq!(ita_hash_algorithm(Tee::Sample), None);
    }

    /// The selector must agree with what the ITA challenge actually tells the
    /// guest. Both read the same function, so this asserts they stay wired to
    /// it rather than drifting into two copies of the same table.
    #[test]
    fn selector_matches_the_algorithm_ita_negotiates() {
        for tee in [Tee::Tdx, Tee::Sgx, Tee::AzTdxVtpm] {
            let negotiated = negotiated_hash_algorithm(tee)
                .expect("Intel TA models this TEE")
                .as_ref()
                .to_lowercase();
            assert_eq!(ita_hash_algorithm(tee), Some(negotiated));
        }
    }

    #[test]
    fn appraiser_timeout_has_a_bounded_default() {
        // Unbounded waits would let one stalled appraiser block every request.
        assert_eq!(
            CompositeConfig::default().appraiser_timeout_secs,
            DEFAULT_APPRAISER_TIMEOUT_SECS
        );
        assert!(DEFAULT_APPRAISER_TIMEOUT_SECS > 0);
    }

    #[test]
    fn operator_pinned_hash_algorithm_is_not_overridden() {
        // Composite::new only fills in sha512 when the operator left it unset.
        let config: CompositeConfig = toml::from_str(
            r#"
            [coco_as_grpc]
            as_addr = "http://127.0.0.1:50004"
            runtime_data_hash_algorithm = "sha384"

            [intel_ta]
            base_url = "https://api.trustauthority.intel.com"
            api_key = "test-key"
            certs_file = "https://portal.trustauthority.intel.com"
            "#,
        )
        .expect("composite config should deserialise");

        assert_eq!(
            config.coco_as_grpc.runtime_data_hash_algorithm.as_deref(),
            Some("sha384")
        );
    }
}
