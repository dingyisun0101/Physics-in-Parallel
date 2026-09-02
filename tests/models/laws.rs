use physics_in_parallel::prelude::models::{PowerLawDecay, PowerLawError, Spring, SpringLawError};

#[test]
fn spring_law_validates_payload_parameters() {
    let spring = Spring::new(2.0, 1.5, Some((0.2, 4.0))).unwrap();
    assert_eq!(spring.k, 2.0);
    assert_eq!(spring.l_0, 1.5);
    assert_eq!(spring.cutoff, Some((0.2, 4.0)));

    assert!(matches!(
        Spring::new(f64::NAN, 1.0, None).unwrap_err(),
        SpringLawError::InvalidSpringConstant { k } if k.is_nan()
    ));
    assert_eq!(
        Spring::new(1.0, -1.0, None).unwrap_err(),
        SpringLawError::InvalidRestLength { l_0: -1.0 }
    );
    assert_eq!(
        Spring::new(1.0, 1.0, Some((2.0, 1.0))).unwrap_err(),
        SpringLawError::InvalidCutoff { min: 2.0, max: 1.0 }
    );
}

#[test]
fn power_law_validates_payload_parameters() {
    let law = PowerLawDecay::new(6.0, -2.0, Some((0.1, 9.0))).unwrap();
    assert_eq!(law.k, 6.0);
    assert_eq!(law.alpha, -2.0);
    assert_eq!(law.range, Some((0.1, 9.0)));

    assert!(matches!(
        PowerLawDecay::new(f64::NAN, 1.0, None).unwrap_err(),
        PowerLawError::InvalidStrength { k } if k.is_nan()
    ));
    assert_eq!(
        PowerLawDecay::new(1.0, f64::INFINITY, None).unwrap_err(),
        PowerLawError::InvalidExponent {
            alpha: f64::INFINITY
        }
    );
    assert_eq!(
        PowerLawDecay::new(1.0, -2.0, Some((5.0, 1.0))).unwrap_err(),
        PowerLawError::InvalidRange { min: 5.0, max: 1.0 }
    );
}

#[test]
fn law_serde_roundtrips_and_validates_input() {
    let spring = Spring::new(2.5, 1.25, Some((0.1, 3.0))).unwrap();
    let spring_json = serde_json::to_string(&spring).unwrap();
    assert_eq!(
        serde_json::from_str::<Spring>(&spring_json).unwrap(),
        spring
    );
    assert!(
        serde_json::from_str::<Spring>(r#"{"k":1.0,"l_0":-1.0,"cutoff":null}"#)
            .unwrap_err()
            .to_string()
            .contains("rest length")
    );
    assert!(
        serde_json::from_str::<Spring>(r#"{"k":1.0,"l_0":1.0,"cutoff":null,"extra":0}"#).is_err()
    );

    let law = PowerLawDecay::new(4.0, -2.0, Some((0.2, 8.0))).unwrap();
    let law_json = serde_json::to_string(&law).unwrap();
    assert_eq!(
        serde_json::from_str::<PowerLawDecay>(&law_json).unwrap(),
        law
    );
    assert!(
        serde_json::from_str::<PowerLawDecay>(r#"{"k":1.0,"alpha":2.0,"range":[3.0,1.0]}"#)
            .unwrap_err()
            .to_string()
            .contains("active range")
    );
}

#[test]
fn law_serde_rejects_invalid_values_during_serialization() {
    let invalid_spring = Spring {
        k: 1.0,
        l_0: -1.0,
        cutoff: None,
    };
    assert!(serde_json::to_string(&invalid_spring).is_err());

    let invalid_power_law = PowerLawDecay {
        k: f64::NAN,
        alpha: -2.0,
        range: None,
    };
    assert!(serde_json::to_string(&invalid_power_law).is_err());
}
