#[cfg(crux)]
extern crate crucible;
#[cfg(crux)]
use crucible::*;
fn sum(mut input: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    let mut i = 0;
    let n = input.len();
    while i < n {
        i += 1;
        output.push(69);
    }
    output
}
fn sum_assert_pre_check(mut input: &[u8]) -> bool {
    true
}
fn sum_assert_post_check(mut input: &[u8], r: Vec<u8>) {
    assert!({
        let condition = |i| !((0 <= i) && (i < (r.len() as i128))) || (r[(i as usize)] == 69);
        let mut result = true;
        {
            for arg0 in 0..(100u32 as i128) {
                for arg0 in [arg0, -arg0] {
                    {
                        if condition(arg0) == false {
                            result = false;
                            break;
                        }
                    }
                }
            }
        };
        result
    })
}
fn sum_assert_wrapper(arg0: &[u8]) {
    if sum_assert_pre_check(arg0) {
        let result = sum(arg0);
        sum_assert_post_check(arg0, result);
    };
}
#[cfg_attr(crux, crux::test)]
fn test_sum_assert_wrapper() {
    let symbolic_array_arg0 = <[u8; 1] as Symbolic>::symbolic("arg0");
    let arg0 = crucible::symbolic::prefix(&symbolic_array_arg0[..]);
    sum_assert_wrapper(arg0);
}
