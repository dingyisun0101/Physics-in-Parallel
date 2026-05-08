use physics_in_parallel::models::laws::{PowerLawDecay, PowerLawError, Spring, SpringLawError};

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
