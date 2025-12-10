#[cfg(crux)]
extern crate crucible;
#[cfg(crux)]
use crucible::*;
fn sum(mut input: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    let mut i = 0;
    let n = input.len();
    output.push(70);
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
    });
}
fn sum_assert_wrapper(arg0: &[u8]) {
    if sum_assert_pre_check(arg0) {
        let result = sum(arg0);
        sum_assert_post_check(arg0, result);
    };
}
#[cfg_attr(crux, crux::test)]
fn test_sum_assert_wrapper() {
    let symbolic_array_arg0 = [< u8 as Symbolic > :: symbolic ("S7b226e616d65223a2261726730222c2273796d626f6c5f74797065223a7b2274797065223a224172726179222c22747970655f6e616d65223a227538222c22696e646578223a307d7d") , < u8 as Symbolic > :: symbolic ("S7b226e616d65223a2261726730222c2273796d626f6c5f74797065223a7b2274797065223a224172726179222c22747970655f6e616d65223a227538222c22696e646578223a317d7d") , < u8 as Symbolic > :: symbolic ("S7b226e616d65223a2261726730222c2273796d626f6c5f74797065223a7b2274797065223a224172726179222c22747970655f6e616d65223a227538222c22696e646578223a327d7d") , < u8 as Symbolic > :: symbolic ("S7b226e616d65223a2261726730222c2273796d626f6c5f74797065223a7b2274797065223a224172726179222c22747970655f6e616d65223a227538222c22696e646578223a337d7d") , < u8 as Symbolic > :: symbolic ("S7b226e616d65223a2261726730222c2273796d626f6c5f74797065223a7b2274797065223a224172726179222c22747970655f6e616d65223a227538222c22696e646578223a347d7d") , < u8 as Symbolic > :: symbolic ("S7b226e616d65223a2261726730222c2273796d626f6c5f74797065223a7b2274797065223a224172726179222c22747970655f6e616d65223a227538222c22696e646578223a357d7d") , < u8 as Symbolic > :: symbolic ("S7b226e616d65223a2261726730222c2273796d626f6c5f74797065223a7b2274797065223a224172726179222c22747970655f6e616d65223a227538222c22696e646578223a367d7d") , < u8 as Symbolic > :: symbolic ("S7b226e616d65223a2261726730222c2273796d626f6c5f74797065223a7b2274797065223a224172726179222c22747970655f6e616d65223a227538222c22696e646578223a377d7d")] ;
    let len_arg0 = usize :: symbolic_where ("S7b226e616d65223a2261726730222c2273796d626f6c5f74797065223a7b2274797065223a2241727261794c656e227d7d" , | & n | n < 8usize) ;
    let arg0 = &symbolic_array_arg0[..len_arg0];
    sum_assert_wrapper(arg0);
}
