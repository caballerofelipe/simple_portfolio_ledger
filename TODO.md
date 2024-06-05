# TODO LIST
- Implement operations
    - Maybe the operation column in the ledger should be called differently, perhaps action. The reason being that operations might be the things that are done via the API and they might hace two events happening such as sell (selling and univesting).
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
- Save/Load ledger from file (CSV maybe or Excel or another custom format).
- Maybe the description column is redundant and the name of the operation is enough. Or maybe be more explicit in the description, instead of 'Buy {instrument}' it could be 'Buy {instrument} in exchange for {price_in}'.
- Maybe bring back a column 'Q_price_commission_tax_verification' with a different name and a different purpose. The column was removed to show where there were inconsistencies between a calculated and stated total, now that's not possible since an error is raised if such an inconsistency occurs if the `tolerance_decimals=4,` is surpassed. But since we are now using a tolerance setting to allow small inconsistencies, maybe a new column would be useful to showcase that small inconsistency.
    

# IDEAS
- There should be cost operations, probably with a 'pay_' prefix:
    - pay_deposit
    - pay_withdraw
    - pay_tax
    - pay_transfer
    -account_cost
- Possibly, origin and destination should always be filled. For instance in a sell operation, origin for the sell part would be the instrument itself.