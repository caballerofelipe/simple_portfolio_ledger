# TODO LIST
- Implement operations
- Add a new column for operation id and sub id?
    - For instance, a sell is an univest and a sell, so both ops should have an id
        e.g. 132 and maybe a sub id 1 and 2
        these could go on different columns or in one column (132-1 and 132-2)
    - The function `_add_row()` should be _add_rows (plural) to allow operation tracking and multiple rows would have the same operation id and multiple sub ids.
    - _add_rows should return the operation id (the main one, not sub ids)
- Add validation for _add_rows
- Change column name `stated_total` to `total`, this makes more sense as it will be calculated and not stated.
- In _cols_operation* the grouping should include price_in as the computation of the same instrument with different reference instrument (price_in) wouldn't make sense (i.e. a stock bought in USD and also in CHF wouldn't allow for the computation to be consistent between the two). In the case that would be needed to be done somehow, a conversion would have to happen.
- Deposit/withdraw
    - If there's a cost should be stated as another operation
    - No commission/tax, add another op for that
    - Only an amount should be deposited or withdrew
- In _cols_operation_balance_by_instrument_for_group(), for withdraw and sell there should be a final review. If withdraw/sell is more than what I have deposit should have a negative number to show an over withdraw or sell
- Review _ledger_columns_attrs.
- Create automated unit tests:
    - Test if operations work as expected.
    - Test cols_operation* methods.
    

# IDEAS
- There should be cost operations, probably with a 'pay_' prefix:
    - pay_deposit
    - pay_withdraw
    - pay_tax
    - pay_transfer
    -account_cost
