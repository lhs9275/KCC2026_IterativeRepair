# Case Study: Bug 153 (Apache Commons Math `linearCombination`)

- Failing test (iter 0): `org.apache.commons.math3.util.MathArraysTest::testLinearCombinationWithSingleElementArray`
- Error-aware outcome: SOLVED at iter 2
- Blind-retry outcome: FAILED after 3 iters

## Iter 0 (both strategies, identical)
```java
public static double linearCombination(final double[] a, final double[] b)
        throws DimensionMismatchException {
        final int len = a.length;
        if (len != b.length) {
            throw new DimensionMismatchException(len, b.length);
        }

        final double[] prodHigh = new double[len];
    ...
```
→ 1 test fails: `testLinearCombinationWithSingleElementArray`

## Error-aware iter 1 feedback (typed `test_feedback`)
```
FAILING TEST COUNT: 1
FAILING TEST NAMES:
  org.apache.commons.math3.util.MathArraysTest::testLinearCombinationWithSingleElementArray
...
FIX: Start from the ORIGINAL BUGGY CODE, not your previous rewrite.
Make a MINIMAL semantic fix near the buggy line.
```

## Error-aware iter 1 result (fixed)
```java
public static double linearCombination(final double[] a, final double[] b)
        throws DimensionMismatchException {
        final int len = a.length;
        if (len != b.length) {
            throw new DimensionMismatchException(len, b.length);
        }

        if (len == 1) {
    ...
```
→ PASS ✓  (single-line guard `if (len == 1) return a[0] * b[0];` added)

## Blind-retry iter 2 (last attempt, still failing)
```java
public static double linearCombination(final double[] a, final double[] b)
        throws DimensionMismatchException {
        final int len = a.length;
        if (len != b.length) {
            throw new DimensionMismatchException(len, b.length);
        }

        final double[] prodHigh = new double[len];
    ...
```
→ still fails; blind prompt gave no test-name hint, model never targets the single-element edge case.

**Takeaway**: error-type-aware's `test_feedback` surfaces the failing test's self-describing name (`testLinearCombinationWithSingleElementArray`), which the LLM uses to localize the missing edge case. Blind-retry's generic 'try a different approach' lacks this signal.