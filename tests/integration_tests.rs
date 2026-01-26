use vectordb::{DistanceMetric, distance};

#[test]
fn test_distance_calculations() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    
    // L2 distance
    let l2_dist = distance(&a, &b, DistanceMetric::L2);
    assert!(l2_dist > 0.0);
    
    // Dot product
    let dot_dist = distance(&a, &b, DistanceMetric::DotProduct);
    assert!(dot_dist < 0.0); // Negative because we negate the dot product
    
    // Cosine distance
    let cos_dist = distance(&a, &b, DistanceMetric::Cosine);
    assert!(cos_dist >= 0.0 && cos_dist <= 2.0);
}

#[test]
fn test_identical_vectors() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    
    let l2_dist = distance(&v, &v, DistanceMetric::L2);
    assert_eq!(l2_dist, 0.0);
    
    let cos_dist = distance(&v, &v, DistanceMetric::Cosine);
    assert!(cos_dist.abs() < 1e-6);
}
