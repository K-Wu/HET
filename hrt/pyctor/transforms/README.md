# Future plans
## InterOpSSA Transforms
### Delta Sugarizer/Desugarizer
"""
This file mainly provides the functionality to sugarize, or desugarize, the delta in the inter-op SSA texts. i.e.,

Delta(TYPE, "varname") <=> (TYPE, "varname_delta")
Delta(WeightName, TYPE) <=> (WeightName_delta, TYPE)
"""

Current status: not implemented. Meanwhile, we will use the desugarized version in all inter-op SSA texts.
