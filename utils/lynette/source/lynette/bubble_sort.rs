use vstd::prelude::*;
use vstd::relations::sorted_by;

verus! {

spec fn sorted(arr: Seq<i32>) -> bool {
    forall |i1: int, i2: int| 0 <= i1 < i2 < arr.len() ==> arr[i1] <= arr[i2]
}

fn bubble_sort(arr: &mut Vec<i32>)
ensures
    sorted(arr.view())
{
    let n = arr.len();

    let mut i = 0;
    while i < n
    invariant
        0 <= i <= n,
        arr.len() == n,
        forall |i1: int, i2: int| ((n - i - 1) < i1 < i2 < n) ==> arr[i1] <= arr[i2],
        forall |i1: int, i2: int| (0 <= i1 < (n - i) <= i2 < n) ==> arr[i1] <= arr[i2],
    // ensures i == n
    // decreases n - i
    {
        let mut swapped = false;
        let mut j = 0;
        while j < (n - i - 1)
        invariant
            0 <= j <= (n - i - 1),
            arr.len() == n,
            // all elements below j are smaller then element j
            forall |elem: int| 0 <= elem < j ==> arr[elem] <= arr[j as int],
            // all elements in sorted part are still sorted
            forall |i1: int, i2: int| ((n - i - 1) < i1 < i2 < n) ==> arr[i1] <= arr[i2],
            // all elements below the sorted end are smaller then the sorted end
            forall |i1: int, i2: int| (0 <= i1 < (n - i) <= i2 < n) ==> arr[i1] <= arr[i2],
            !swapped ==> forall |i1: int, i2: int| 0 <= i1 < i2 <= j ==> arr[i1] <= arr[i2],
        // decreases (n - i - 1) - j
        {
            if arr[j] > arr[j + 1] {
                let tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
                swapped = true;
            }

            j += 1;
        }

        // If no swaps happened, the array is already sorted
        if !swapped {
            i = n;
            break;
        }
        i += 1;
    }
}

}
