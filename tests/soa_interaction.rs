use physics_in_parallel::engines::soa::interaction::{
    Interaction, InteractionError, PairTopology,
};
use physics_in_parallel::engines::soa::phys_obj::PhysObj;

#[derive(Debug, Clone)]
struct ConstantPairPush {
    topology: PairTopology,
    gain: f64,
}

impl ConstantPairPush {
    /// Annotation:
    /// - Purpose: Executes `new` logic.
    /// - Parameters:
    ///   - `n_objects` (`usize`): Parameter of type `usize` used by `new`.
    ///   - `gain` (`f64`): Parameter of type `f64` used by `new`.
    fn new(n_objects: usize, gain: f64) -> Self {
        Self {
            topology: PairTopology::new(n_objects),
            gain,
        }
    }
}

impl Interaction<1> for ConstantPairPush {
    /// Annotation:
    /// - Purpose: Executes `topology` logic.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn topology(&self) -> &PairTopology {
        &self.topology
    }

    /// Annotation:
    /// - Purpose: Executes `topology_mut` logic.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn topology_mut(&mut self) -> &mut PairTopology {
        &mut self.topology
    }

    /// Annotation:
    /// - Purpose: Executes `accumulate_acc` logic.
    /// - Parameters:
    ///   - `objects` (`&mut PhysObj<1>`): Parameter of type `&mut PhysObj<1>` used by `accumulate_acc`.
    fn accumulate_acc(&self, objects: &mut PhysObj<1>) -> Result<(), InteractionError> {
        for &e in self.topology.active_edges() {
            let (i, j) = self.topology.edge_pair(e)?;

            if !objects.is_alive(i)? || !objects.is_alive(j)? {
                continue;
            }

            objects.acc_of_mut(i)?[0] += self.gain;
            objects.acc_of_mut(j)?[0] -= self.gain;
        }
        Ok(())
    }
}

#[test]
/// Annotation:
/// - Purpose: Executes `pair_topology_insert_delete_and_adj_matrix` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn pair_topology_insert_delete_and_adj_matrix() {
    let mut topo = PairTopology::new(4);

    let e01 = topo.insert(0, 1).unwrap();
    let e23 = topo.insert(2, 3).unwrap();
    let e12 = topo.insert(1, 2).unwrap();

    assert_eq!(topo.active_count(), 3);
    assert!(topo.contains_pair(1, 0));
    assert_eq!(topo.index_of(0, 1), Some(e01));
    assert_eq!(topo.index_of(3, 2), Some(e23));
    assert_eq!(topo.index_of(2, 1), Some(e12));

    let a = topo.to_adj_matrix();
    assert_eq!(a.get(0, 1), 1);
    assert_eq!(a.get(1, 0), 1);
    assert_eq!(a.get(2, 3), 1);
    assert_eq!(a.get(3, 2), 1);
    assert_eq!(a.get(1, 2), 1);
    assert_eq!(a.get(2, 1), 1);
    assert_eq!(a.get(0, 0), 0);
    assert_eq!(a.get(3, 3), 0);
    assert_eq!(a.get(0, 3), 0);

    topo.delete(2, 3).unwrap();
    assert_eq!(topo.active_count(), 2);
    assert!(!topo.contains_pair(2, 3));
    assert!(!topo.is_active_index(e23));
}

#[test]
/// Annotation:
/// - Purpose: Executes `pair_topology_rejects_invalid_edges` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn pair_topology_rejects_invalid_edges() {
    let mut topo = PairTopology::new(3);

    assert_eq!(
        topo.insert(1, 1),
        Err(InteractionError::SelfEdge { obj: 1 })
    );

    assert_eq!(
        topo.insert(0, 9),
        Err(InteractionError::InvalidObjId {
            obj: 9,
            n_objects: 3
        })
    );
}

#[test]
/// Annotation:
/// - Purpose: Executes `interaction_trait_step_accumulates_through_topology` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn interaction_trait_step_accumulates_through_topology() {
    let mut obj = PhysObj::<1>::empty(2);
    obj.clear_acc();

    let mut inter = ConstantPairPush::new(2, 2.5);
    inter.topology_mut().insert(0, 1).unwrap();

    inter.step(&mut obj).unwrap();

    assert_eq!(obj.acc_of(0).unwrap(), [2.5].as_slice());
    assert_eq!(obj.acc_of(1).unwrap(), [-2.5].as_slice());
}

#[test]
/// Annotation:
/// - Purpose: Executes `interaction_trait_skips_dead_objects_when_law_chooses_to` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn interaction_trait_skips_dead_objects_when_law_chooses_to() {
    let mut obj = PhysObj::<1>::empty(2);
    obj.clear_acc();
    obj.kill_object(1).unwrap();

    let mut inter = ConstantPairPush::new(2, 1.0);
    inter.topology_mut().insert(0, 1).unwrap();

    inter.step(&mut obj).unwrap();

    assert_eq!(obj.acc_of(0).unwrap(), [0.0].as_slice());
    assert_eq!(obj.acc_of(1).unwrap(), [0.0].as_slice());
}
