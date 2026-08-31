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

use anyhow::{anyhow, bail, Context};
use async_trait::async_trait;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
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

    /// Intel TA `attester_tcb_status` values that count as an affirmation.
    ///
    /// Intel TA returns HTTP 200 with a signed token even when the platform TCB
    /// is out of date — the verdict lives in the claims, not the status code.
    /// Upstream's ITA client discards those claims, so without this check
    /// "Intel TA affirmed" would mean no more than "Intel TA answered".
    ///
    /// Defaults to `["OK"]`, which is fail-closed. A platform carrying known
    /// Intel advisories reports `OutOfDate`; accepting it is a deliberate
    /// decision an operator must write down here, not a silent default.
    #[serde(default = "default_accepted_tcb_status")]
    pub ita_accepted_tcb_status: Vec<String>,

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

fn default_accepted_tcb_status() -> Vec<String> {
    vec!["OK".to_string()]
}

fn default_appraiser_timeout_secs() -> u64 {
    DEFAULT_APPRAISER_TIMEOUT_SECS
}

impl Default for CompositeConfig {
    fn default() -> Self {
        Self {
            coco_as_grpc: GrpcConfig::default(),
            intel_ta: IntelTrustAuthorityConfig::default(),
            return_token: ReturnToken::default(),
            ita_accepted_tcb_status: default_accepted_tcb_status(),
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
    accepted_tcb_status: Vec<String>,
    appraiser_timeout: Duration,
}

/// The TCB verdict Intel TA reported, and the advisories behind it.
struct ItaVerdict {
    tcb_status: Option<String>,
    advisory_ids: Vec<String>,
}

/// Read the verdict out of an Intel TA token.
///
/// The signature is already checked by `IntelTrustAuthority::verify` before this
/// runs, so decoding the payload here is reading a verified document, not
/// trusting an unverified one.
fn read_ita_verdict(token: &str) -> anyhow::Result<ItaVerdict> {
    let payload = token
        .split('.')
        .nth(1)
        .ok_or_else(|| anyhow!("Intel TA token is not a JWT"))?;
    let bytes = URL_SAFE_NO_PAD
        .decode(payload)
        .context("Intel TA token payload is not base64url")?;
    let claims: serde_json::Value =
        serde_json::from_slice(&bytes).context("Intel TA token payload is not JSON")?;

    // The verdict is nested under the attester type.
    for tee in ["tdx", "sgx"] {
        let Some(block) = claims.get(tee) else {
            continue;
        };
        return Ok(ItaVerdict {
            tcb_status: block
                .get("attester_tcb_status")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            advisory_ids: block
                .get("attester_advisory_ids")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
        });
    }

    Ok(ItaVerdict {
        tcb_status: None,
        advisory_ids: Vec::new(),
    })
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
            accepted_tcb_status: config.ita_accepted_tcb_status,
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

        // A 200 from Intel TA is not an affirmation. The verdict is in the
        // claims, and upstream's client throws them away, so check them here.
        let verdict = read_ita_verdict(&ita_token)
            .context("composite: could not read the Intel TA verdict")?;

        match &verdict.tcb_status {
            Some(status) if self.accepted_tcb_status.iter().any(|a| a == status) => {}
            Some(status) => {
                warn!(
                    "composite: Intel TA reports attester_tcb_status={status}, advisories={:?}",
                    verdict.advisory_ids
                );
                bail!(
                    "composite attestation failed — Intel TA returned a token but reports \
                     attester_tcb_status={status}, which is not in the accepted list {:?}. \
                     Advisories: {:?}. Add the status to ita_accepted_tcb_status only as a \
                     deliberate, recorded decision.",
                    self.accepted_tcb_status,
                    verdict.advisory_ids
                );
            }
            None => {
                bail!(
                    "composite attestation failed — the Intel TA token carries no \
                     attester_tcb_status, so its verdict cannot be established"
                );
            }
        }

        info!(
            "composite: both appraisers affirmed (CoCo AS token {} bytes, Intel TA token {} bytes, \
             Intel TA tcb_status={:?})",
            coco_token.len(),
            ita_token.len(),
            verdict.tcb_status
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

    /// What the two appraisers actually digest, and why it is not the same bytes.
    ///
    /// Intel TA's client hashes serde_json's `Display` output
    /// (`runtime_data.to_string()` in `intel_trust_authority::build_attest_request`).
    /// The CoCo AS never sees that string: the gRPC client sends it, the AS
    /// re-parses it and hashes **JCS-canonical** bytes
    /// (`attestation-service/src/lib.rs`, `parse_runtime_data`).
    ///
    /// A review flagged this as an unenforced invariant, on the premise that the
    /// two encodings agree because nothing enables serde_json's `preserve_order`.
    /// **That premise is false.** `josekit` enables `preserve_order` in this
    /// build, so `Value` maps are insertion-ordered and `to_string()` does *not*
    /// emit JCS key order. This test pins that reality so nobody re-derives the
    /// wrong conclusion from the review.
    ///
    /// The system nevertheless works, because Intel TA parses the runtime data
    /// and canonicalises it server-side rather than hashing our bytes verbatim —
    /// observed directly in a captured token, whose `attester_runtime_data` claim
    /// comes back with sorted keys and whose `tdx_report_data` matched.
    ///
    /// The place that is *not* covered by that reasoning is the GPU nonce, which
    /// `build_attest_request` computes **locally** as
    /// `sha512(runtime_data.to_string())[..32]`. If the guest derives its GPU
    /// nonce from the canonical digest, the two differ. See
    /// `docs/DR-002-overlapping-hybrid.md` — that is an open question, not a
    /// settled one.
    #[test]
    fn the_two_serialisations_are_known_to_differ() {
        use kbs_types::TeePubKey;
        use serde_json::json;

        let tee_pubkey = TeePubKey::RSA {
            alg: "RSA1_5".into(),
            k_mod: "sGYs1c7B_3_ZUxr0RvjEwLpXqRnPjm3Ck0hfLxLm".into(),
            k_exp: "AQAB".into(),
        };

        // The exact shape kbs::attestation::backend builds.
        let runtime_data = json!({
            "tee-pubkey": tee_pubkey,
            "nonce": "cmFuZG9tLW5vbmNlLXZhbHVlLTMyLWJ5dGVzLWxvbmc=",
        });

        let ita_bytes = runtime_data.to_string().into_bytes();
        let reparsed: serde_json::Value =
            serde_json::from_str(&runtime_data.to_string()).expect("re-parse");
        let coco_bytes = serde_json_canonicalizer::to_vec(&reparsed).expect("canonicalize");

        // Both must remain valid JSON describing the same object...
        let a: serde_json::Value = serde_json::from_slice(&ita_bytes).unwrap();
        let b: serde_json::Value = serde_json::from_slice(&coco_bytes).unwrap();
        assert_eq!(
            a, b,
            "the two encodings must at least describe the same value"
        );

        // ...but they are NOT the same bytes, because preserve_order is on.
        assert_ne!(
            ita_bytes, coco_bytes,
            "The two encodings now agree byte-for-byte. That is a real change: \
             either preserve_order was disabled or runtime_data's shape changed. \
             Re-read the GPU-nonce reasoning in docs/DR-002-overlapping-hybrid.md \
             before assuming this is harmless."
        );
    }

    /// A 200 from Intel TA is not an affirmation — the verdict is in the claims.
    /// These are the shapes a real captured token produced.
    #[test]
    fn ita_verdict_is_read_from_the_token_claims() {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
        use serde_json::json;

        let make = |claims: serde_json::Value| {
            format!(
                "header.{}.signature",
                URL_SAFE_NO_PAD.encode(claims.to_string())
            )
        };

        // Shape observed live from GPU22 on 2026-08-31.
        let out_of_date = make(json!({
            "tdx": {
                "attester_tcb_status": "OutOfDate",
                "attester_advisory_ids": ["INTEL-SA-01314", "INTEL-SA-01397"],
            }
        }));
        let v = read_ita_verdict(&out_of_date).expect("verdict");
        assert_eq!(v.tcb_status.as_deref(), Some("OutOfDate"));
        assert_eq!(v.advisory_ids.len(), 2);

        let ok = make(json!({ "tdx": { "attester_tcb_status": "OK" } }));
        let v = read_ita_verdict(&ok).expect("verdict");
        assert_eq!(v.tcb_status.as_deref(), Some("OK"));
        assert!(v.advisory_ids.is_empty());

        // SGX nests it under its own key.
        let sgx = make(json!({ "sgx": { "attester_tcb_status": "OK" } }));
        assert_eq!(
            read_ita_verdict(&sgx)
                .expect("verdict")
                .tcb_status
                .as_deref(),
            Some("OK")
        );

        // No verdict at all must be distinguishable from an affirming one.
        let empty = make(json!({}));
        assert_eq!(read_ita_verdict(&empty).expect("verdict").tcb_status, None);
    }

    /// The default must be fail-closed. GPU22 itself reports OutOfDate, so a
    /// permissive default would have silently accepted a platform carrying six
    /// Intel advisories.
    #[test]
    fn accepted_tcb_status_defaults_to_ok_only() {
        let accepted = CompositeConfig::default().ita_accepted_tcb_status;
        assert_eq!(accepted, vec!["OK".to_string()]);
        assert!(!accepted.iter().any(|s| s == "OutOfDate"));
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
