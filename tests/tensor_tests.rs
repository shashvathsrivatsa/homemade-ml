use homemade_ml::{TensorNode, matmul};

#[test]
fn matmul_multiplies_square_matrices() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

    let a = TensorNode::new(a.into_iter().flatten().collect(), vec![2, 2]);
    let b = TensorNode::new(b.into_iter().flatten().collect(), vec![2, 2]);

    let result = matmul(&a, &b);

    assert_eq!(result.get(&[0, 0]), 19.0);
    assert_eq!(result.get(&[0, 1]), 22.0);
    assert_eq!(result.get(&[1, 0]), 43.0);
    assert_eq!(result.get(&[1, 1]), 50.0);
}

#[test]
fn matmul_multiplies_rectangular_matrices() {
    let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let b = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];

    let a = TensorNode::new(a.into_iter().flatten().collect(), vec![2, 3]);
    let b = TensorNode::new(b.into_iter().flatten().collect(), vec![3, 2]);

    let result = matmul(&a, &b);

    assert_eq!(result.get(&[0, 0]), 58.0);
    assert_eq!(result.get(&[0, 1]), 64.0);
    assert_eq!(result.get(&[1, 0]), 139.0);
    assert_eq!(result.get(&[1, 1]), 154.0);
}

#[test]
fn matmul_with_identity_returns_original_matrix() {
    let matrix = vec![vec![2.5, -1.0], vec![0.0, 4.0], vec![3.0, 7.5]];
    let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    let matrix = TensorNode::new(matrix.into_iter().flatten().collect(), vec![3, 2]);
    let identity = TensorNode::new(identity.into_iter().flatten().collect(), vec![2, 2]);

    let result = matmul(&matrix, &identity);

    assert_eq!(result.get(&[0, 0]), 2.5);
    assert_eq!(result.get(&[0, 1]), -1.0);
    assert_eq!(result.get(&[1, 0]), 0.0);
    assert_eq!(result.get(&[1, 1]), 4.0);
    assert_eq!(result.get(&[2, 0]), 3.0);
    assert_eq!(result.get(&[2, 1]), 7.5);
}

#[test]
#[should_panic(expected = "invalid dimensions for matmul")]
fn matmul_rejects_incompatible_dimensions() {
    let a = vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]];
    let b = vec![
        vec![1.0, 1.0],
        vec![1.0, 1.0],
        vec![1.0, 1.0],
        vec![1.0, 1.0],
    ];

    let a = TensorNode::new(a.into_iter().flatten().collect(), vec![2, 3]);
    let b = TensorNode::new(b.into_iter().flatten().collect(), vec![4, 2]);

    let _ = matmul(&a, &b);
}
