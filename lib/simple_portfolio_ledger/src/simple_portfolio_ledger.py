'''A simple ledger to keep track of a Portfolio's movements.'''

import functools
import inspect
import warnings
from typing import List

import numpy as np
import pandas as pd

'''
Notes:
- MARK is a functionality from Visual Studio Code to find things in the minimap.
'''

# MARK: TODO LIST
# TODO
_ = '''
- For all operations
    - Add type validation
- Add a new column for operation id and sub id?
    - For instance, a sell is an univest and a sell, so both ops should have an id
        e.g. 132 and maybe a sub id 1 and 2
        these could go on different columns or in one column (132-1 and 132-2)
    - The function `_add_row()` should be _add_rows (plural) to allow operation tracking and multiple rows would have the same operation id and multiple sub ids.
- In _cols_operation* the grouping should include price_in as the computation of the same instrument with different reference instrument (price_in) wouldn't make sense (i.e. a stock bought in USD and also in CHF wouldn't allow for the computation to be consistent between the two). In the case that would be needed to be done somehow, a conversion would have to happen.
- Deposit/withdraw
    - If there's a cost should be stated as another operation
    - No commission/tax, add another op for that
    - Only an amount should be deposited or withdrew

- Review _ledger_columns_attrs.
- In _cols_operation_balance_by_instrument_for_group(), for withdraw and sell
    there should be a final review. If withdraw/sell is more than what I have
    deposit should have a negative number to show an over withdraw or sell
IDEAS: 
- There should be cost operations, probably with a 'pay_' prefix:
    - pay_deposit
    - pay_withdraw
    - pay_tax
    - pay_transfer
    - account_cost
'''


class SimplePortfolioLedger:

    _ledger_columns = (
        'date_execution',
        'operation',
        'instrument',
        'origin',
        'destination',
        'price_in',
        'price',
        'price_w_expenses',
        'size',
        'commission',
        'tax',
        'stated_total',
        'date_order',
        'description',
        'notes',
        'commission_notes',
        'tax_notes',
        'account',
        'Q_price_commission_tax_verification',
    )

    # Column description in a dict
    _ledger_columns_attrs = {
        'column notes': {
            'date_execution': 'The date the transaction was actually done.',
            'date_order': 'The date the transaction was created.',
            #
            # Ideas for operation:
            #   - reinvest (uninvest and invest in something else)
            #   - invest: change something into something else
            #   - uninvest:
            #
            'operation': 'Can be: buy, sell, deposit (for account), withdraw (for account)',
            'instrument': 'Instrument code (USD, CLP, COP, AAPL, AMZN).',
            'origin': 'Where did the resource come from.',
            'destination': 'Where did the resource go.',
            'price_in': 'Instrument used for the operation.',
            'price': 'Price',
            'size': 'The amount in the operation.',
            'commission': '',
            'tax': '',
            'commission_notes' 'tax_notes': '',
            # Used to store additional information
            'description': 'What happened.',
            'notes': 'Additional notes for the transaction.',
            # These columns would allow me to have just one file for all my investments
            #   but might make the process more cumbersome
            'account': 'Account where the transaction was done.',
            # Debugging
            'Q_price_commission_tax_verification': '',
        }
    }

    _ops_names = set(
        (
            'buy',
            'deposit',
            'dividend',
            'invest',
            'sell',
            'stock dividend',
            'uninvest',
            'withdraw',
        )
    )

    def __init__(self) -> None:
        self._ledger_df = self._create_empty_ledger_df()
        self._instruments_metadata = {}

    # *****************
    # Decorators
    # MARK: Decorators
    # *****************

    @staticmethod
    def _deco_check_ledger_for_cols(func):
        # https://stackoverflow.com/a/72563047/1071459
        deco_name = inspect.stack()[0].function

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            if len(self._ledger_df) == 0:
                warnings.warn(
                    'WARNING: Ledger is empty, no columns computed,'
                    + ' showing only basic column structure.'
                    + f' (Warning issued by decorator [{deco_name}])',
                    stacklevel=2,
                )

            extraneous_ops = self._get_extraneous_ops()
            if len(extraneous_ops) > 0:
                raise ValueError(
                    'One or more forbidden operations were inserted in the ledger.',
                    extraneous_ops,
                )

            return func(self, *args, **kwargs)

        return wrapper

    # *****************
    # Classmethods
    # MARK: Classmethods
    # *****************

    @classmethod
    def whatis(cls, columns):
        """Return information about different parts of the data."""
        if isinstance(columns, list):
            to_return = []
            for col in columns:
                to_return.append(
                    {col: cls._ledger_columns_attrs['column notes'].get(col, 'NOT DEFINED')}
                )
            return to_return
        elif isinstance(columns, str):
            return cls._ledger_columns_attrs['column notes'].get(columns, 'NOT DEFINED')
        else:
            warnings.warn('Columns must be list or str. Returning None.')
            return None

    @classmethod
    def _create_empty_ledger_df(cls) -> pd.DataFrame:
        """Returns an empty ledger DataFrame.

        Returns
        -------
        pd.DataFrame
            Empty ledger DataFrame.
        """
        # Create an empty portfolio_ledger
        ledger = pd.DataFrame(columns=cls._ledger_columns)

        # About attrs:
        #  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.attrs.html
        ledger.attrs = cls._ledger_columns_attrs

        return ledger

    def _get_extraneous_ops(self):
        # Compute extraneous_ops, which shows all unique operations from self._ledger_df
        #  minus all operations self._ops_names
        #  leaving operations that shouldn't be here
        used_ops = set(self._ledger_df['operation'].unique())
        extraneous_ops = used_ops - set(self._ops_names)
        return extraneous_ops

    # *****************
    # Computing operations
    # MARK: Computing Operations
    # *****************

    def ledger(
        self,
        instrument_type: bool = False,
        instrument_name: bool = False,
        cols_operation: bool = False,
        cols_operation_cumsum: bool = False,
        cols_operation_balance_by_instrument: bool = False,
        thousands_fmt_sep=False,
        thousands_fmt_decimals=1,
        simplify_dtypes=True,
    ) -> pd.DataFrame:
        """Returns The Ledger, with optional additional information.

        Parameters
        ----------
        instrument_type : bool, optional
            Wether to add a column for the instrument type. By default False.
        instrument_name : bool, optional
            Wether to add a column for the instrument name. By default False.
        cols_operation : bool, optional
            Whether to add cols_operation to The Ledger. By default False.
        cols_operation_cumsum : bool, optional
            Whether to add cols_operation_cumsum to The Ledger. By default False.
        cols_operation_balance_by_instrument : bool, optional
            Whether to add cols_operation_balance_by_instrument to The Ledger. By default False.
        thousands_fmt_sep : bool, optional
            Add a thousands separator. By default False.
        thousands_fmt_decimals : int, optional
            Decimals to print, used only when thousands_fmt_sep is set to True. By default 1.
        simplify_dtypes : bool, optional
             Allows to simplify dtypes, for instance, pass from float64 to int64 if no decimals are present. Doesn't convert to a dtype that supports pd.NA, like `DataFrame.convert_dtypes()` although it uses it. See https://github.com/pandas-dev/pandas/issues/58543#issuecomment-2101240339 . Warning: Might have a performance impact if True. By default True.

        Returns
        -------
        pd.DataFrame
            The Ledger, with optional additional information.
        """
        if len(self._ledger_df) == 0:
            warnings.warn('WARNING: Ledger is empty, showing only basic structure.')

        # Specifying columns to avoid returning manually added columns
        # `*` needed because self._ledger_columns is a tuple
        the_ledger = self._ledger_df[[*self._ledger_columns]]

        if instrument_type is True or instrument_name is True:
            instruments = self.instruments(
                instrument_type=instrument_type,
                instrument_name=instrument_name,
                instrument_in_ledger=False,
            )
            the_ledger = pd.merge(the_ledger, instruments, how='left', on='instrument')

        if simplify_dtypes is True:
            with pd.option_context('future.no_silent_downcasting', True):
                the_ledger = (
                    the_ledger
                    # See https://github.com/pandas-dev/pandas/issues/58543#issuecomment-2101240339
                    .astype('object')
                    .convert_dtypes()
                    .astype('object')
                    .replace(pd.NA, float('nan'))
                    .infer_objects()
                )

        dfs_toconcat = [the_ledger]

        if cols_operation is True:
            tmp = self.cols_operation(simplify_dtypes=simplify_dtypes)
            dfs_toconcat.append(tmp)
        if cols_operation_cumsum is True:
            tmp = self.cols_operation_cumsum(simplify_dtypes=simplify_dtypes)
            dfs_toconcat.append(tmp)
        if cols_operation_balance_by_instrument is True:
            tmp = self.cols_operation_balance_by_instrument(simplify_dtypes=simplify_dtypes)
            dfs_toconcat.append(tmp)

        to_return = (
            pd
            # Join DataFrames
            .concat(dfs_toconcat, join='inner', axis=1)
            # Convert dates to datetime64[s]
            .astype(
                {
                    'date_execution': 'datetime64[s]',
                    'date_order': 'datetime64[s]',
                }
            )
            # Try to pass colums where dtype is object to a type like int64 or float64
            .infer_objects()
        )

        if thousands_fmt_sep is True:
            return (
                to_return
                # On floats, add thousands separator and [thousands_fmt_decimals] decimals
                .map(lambda x: (f'{x:,.{thousands_fmt_decimals}f}' if isinstance(x, float) else x))
                # On ints, add thousands separator
                .map(lambda x: f'{x:,d}' if isinstance(x, int) else x)
            )
        else:
            return to_return

    @_deco_check_ledger_for_cols
    def cols_operation(self, show_instr_accnt=False, simplify_dtypes=True) -> pd.DataFrame:
        """Returns a dataframe with 1 column per operation.

        Parameters
        ----------
        show_instr_accnt : bool, optional
            Whether or not to show the instrument and the account. By default False.
        simplify_dtypes : bool, optional
            Allows to simplify dtypes, for instance, pass from float64 to int64 if no decimals are present. Doesn't convert to a dtype that supports pd.NA, like `DataFrame.convert_dtypes()` although it uses it. See https://github.com/pandas-dev/pandas/issues/58543#issuecomment-2101240339 . Warning: Might have a performance impact if True. By default True.

        Returns
        -------
        pd.DataFrame
            A dataframe with 1 column per operation.
        """

        if len(self._ledger_df) == 0:
            # Return an empty DataFrame with whe structure needed
            to_return = pd.DataFrame(columns=['instrument', 'account', *sorted(self._ops_names)])
        else:
            # Create groups by instrument/operation
            groups = self._ledger_df.groupby(['instrument', 'account', 'operation'])

            # For each group, return the size
            #  (for example if an op is buy, the column buy would be filled, not all others)
            op_size = groups.apply(lambda x: x['size'], include_groups=False)

            with pd.option_context('future.no_silent_downcasting', True):
                to_return = (
                    op_size
                    # Move operation from the index and create one column for each op
                    .unstack('operation')
                    # Fill na with 0
                    .fillna(0)
                    # Add columns that weren't created in the previous operation
                    #   using the operation list from the class self._ops_names
                    .reindex(sorted(self._ops_names), axis=1, fill_value=0)
                    # Move index instrument created by the last groupby to a column
                    .reset_index(['instrument', 'account'])
                    # Sort by the original index
                    .sort_index()
                    # Try to pass colums where dtype is object to a type like int64 or float64
                    .infer_objects()
                )

                if simplify_dtypes is True:
                    with pd.option_context('future.no_silent_downcasting', True):
                        to_return = (
                            to_return
                            # See https://github.com/pandas-dev/pandas/issues/58543#issuecomment-2101240339
                            .astype('object')
                            .convert_dtypes()
                            .astype('object')
                            .replace(pd.NA, float('nan'))
                            .infer_objects()
                        )

        if show_instr_accnt is True:
            return to_return[['instrument', 'account', *sorted(self._ops_names)]]
        else:
            return to_return[[*sorted(self._ops_names)]]

    @_deco_check_ledger_for_cols
    def cols_operation_cumsum(self, show_instr_accnt=False, simplify_dtypes=True) -> pd.DataFrame:
        """Returns a DataFrame with one column per operation but do a cumsum per instrument/account.

        Parameters
        ----------
        show_instr_accnt : bool, optional
            Whether or not to show the instrument and the account. By default False.
        simplify_dtypes : bool, optional
            Allows to simplify dtypes, for instance, pass from float64 to int64 if no decimals are present. Doesn't convert to a dtype that supports pd.NA, like `DataFrame.convert_dtypes()` although it uses it. See https://github.com/pandas-dev/pandas/issues/58543#issuecomment-2101240339 . Warning: Might have a performance impact if True. By default True.

        Returns
        -------
        pd.DataFrame
            A DataFrame with one column per operation but do a cumsum per instrument/operation/account.
        """

        # List of columns to return, SORTED by self._ops_names
        ops_cumsum_names = [f'cumsum {c}' for c in sorted(self._ops_names)]

        if len(self._ledger_df) == 0:
            # Return an empty DataFrame with whe structure needed
            to_return = pd.DataFrame(
                columns=[
                    ['instrument', 'account', *ops_cumsum_names, 'cumsum held', 'cumsum invested']
                ]
            )
        else:
            # Create groups by instrument/operation
            groups = self._ledger_df.groupby(['operation', 'instrument', 'account'])

            # For each group do a cumsum for size
            op_size_cumsum = groups.apply(lambda x: x['size'].cumsum(), include_groups=False)

            # Used to rename the columns to their appropriate name
            # WARNING: `sorted(self._ops_names)` must be sorted because
            #  ops_cumsum_names is sorted by self._ops_names
            #  if not sorted columns and column names won't match
            rename_dict = dict(zip(sorted(self._ops_names), ops_cumsum_names))

            with pd.option_context('future.no_silent_downcasting', True):
                to_return = (
                    op_size_cumsum
                    # Move operation from the index and create one column for each op
                    .unstack('operation')
                    # Add columns that weren't created in the previous operation
                    #  using the operation list from the class' self._ops_names
                    .reindex(sorted(self._ops_names), axis=1, fill_value=0)
                    # Rename created columns
                    .rename(columns=rename_dict)
                    # Move indexes 'instrument' and 'account'created by the last groupby to a column
                    .reset_index(['instrument', 'account'])
                    # Regroup by instrument to do the ffill
                    .groupby(['instrument', 'account'])
                    # For each instrument, ffill with the cumsum, fill with 0 the beginning
                    .apply(lambda x: x.ffill().fillna(0), include_groups=False)
                    # Move indexes 'instrument' and 'account'created by the last groupby to a column
                    .reset_index(['instrument', 'account'])
                    # Create new colums
                    .assign(
                        # sum of all cumsum columns
                        held=lambda x: x[ops_cumsum_names].sum(axis=1),
                        # Invested total in positive value
                        invested=lambda x: x[['cumsum invest', 'cumsum uninvest']]
                        .sum(axis=1)
                        .abs(),
                    )
                    # Rename newly created columns
                    .rename(columns={'held': 'cumsum held', 'invested': 'cumsum invested'})
                    # Sort by the original index
                    .sort_index()
                    # Try to pass colums where dtype is object to a type like int64 or float64
                    .infer_objects()
                )

                if simplify_dtypes is True:
                    with pd.option_context('future.no_silent_downcasting', True):
                        to_return = (
                            to_return
                            # See https://github.com/pandas-dev/pandas/issues/58543#issuecomment-2101240339
                            .astype('object')
                            .convert_dtypes()
                            .astype('object')
                            .replace(pd.NA, float('nan'))
                            .infer_objects()
                        )

        if show_instr_accnt is True:
            return to_return[
                ['instrument', 'account', *ops_cumsum_names, 'cumsum held', 'cumsum invested']
            ]
        else:
            return to_return[[*ops_cumsum_names, 'cumsum held', 'cumsum invested']]

    @staticmethod
    def _cols_operation_balance_by_instrument_for_group(group_df, new_columns) -> pd.DataFrame:
        """
        WARNING: not to be called by itself. It needs a grouping per instrument.
        """

        # Copy df so that no unexpected changes occur due to pointing logic
        df = group_df.copy()

        # Add new columns
        #  Must be set to 0.0 instead of 0 because if set to 0 and then a
        #   decimal is added to a column it will create an incompatible dtype warning
        df[new_columns] = 0.0

        # float('nan') is similar to np.nan, could check with np.isnan
        #  np.isnan(float('nan')) is True
        df['avg buy total price'] = float('nan')

        # Cols to pass from previous row to current
        cols_to_copy = [col for col in new_columns if 'balance' in col]
        # print(cols_to_copy)

        # Initially prev index is the same as current index
        prev_idx = df.index[0]
        # print(df.loc[prev_idx, cols_to_copy])

        for row in df.iterrows():
            idx, info = row

            # Copy previous operation (index: prev_idx) balance
            #  (in this loop the current operation (index: idx) will be changed)
            #  This is necessary as some operations don't touch every new column
            #  and this assures the untouched operations keep track of history
            df.loc[idx, cols_to_copy] = df.loc[prev_idx, cols_to_copy]

            # invest and uninvest should not be part of the balance
            #  the invested balance is computed in `cols_operation_cumsum`, this applies to:
            # `info['operation'] == 'invest'` and `info['operation'] == 'uninvest'`

            if info['operation'] == 'deposit':
                df.loc[idx, 'balance deposit'] = (
                    df.loc[prev_idx, 'balance deposit'] + df.loc[idx, 'size']
                )

            elif info['operation'] == 'buy':
                df.loc[idx, 'balance buy'] = df.loc[prev_idx, 'balance buy'] + df.loc[idx, 'size']
                df.loc[idx, 'balance buy total payed'] = (
                    df.loc[prev_idx, 'balance buy total payed'] + df.loc[idx, 'stated_total']
                )
                # df.loc[idx, 'avg buy total price'] will be computed at the end
                #  of each iteration of this for

            elif info['operation'] == 'dividend':
                df.loc[idx, 'balance dividend'] = (
                    df.loc[prev_idx, 'balance dividend'] + df.loc[idx, 'size']
                )

            elif info['operation'] == 'stock dividend':
                df.loc[idx, 'balance stock dividend'] = (
                    df.loc[prev_idx, 'balance stock dividend'] + df.loc[idx, 'size']
                )

            elif info['operation'] == 'withdraw':
                # withdraw takes money if available in this order from:
                #  [deposit, stock dividend, dividend, buy]
                # withdraw is negative, use absolute
                withdrew = abs(df.loc[idx, 'size'])

                if withdrew > 0 and df.loc[prev_idx, 'balance deposit'] > 0:
                    if withdrew > df.loc[prev_idx, 'balance deposit']:
                        withdrew = withdrew - df.loc[prev_idx, 'balance deposit']
                        df.loc[idx, 'balance deposit'] = 0
                    else:
                        df.loc[idx, 'balance deposit'] = (
                            df.loc[prev_idx, 'balance deposit'] - withdrew
                        )
                        withdrew = 0
                if withdrew > 0 and df.loc[prev_idx, 'balance stock dividend'] > 0:
                    if withdrew > df.loc[prev_idx, 'balance stock dividend']:
                        withdrew = withdrew - df.loc[prev_idx, 'balance stock dividend']
                        df.loc[idx, 'balance stock dividend'] = 0
                    else:
                        df.loc[idx, 'balance stock dividend'] = (
                            df.loc[prev_idx, 'balance stock dividend'] - withdrew
                        )
                        withdrew = 0
                if withdrew > 0 and df.loc[prev_idx, 'balance dividend'] > 0:
                    if withdrew > df.loc[prev_idx, 'balance dividend']:
                        withdrew = withdrew - df.loc[prev_idx, 'balance dividend']
                        df.loc[idx, 'balance dividend'] = 0
                    else:
                        df.loc[idx, 'balance dividend'] = (
                            df.loc[prev_idx, 'balance dividend'] - withdrew
                        )
                        withdrew = 0
                if withdrew > 0:
                    df.loc[idx, 'balance buy'] = df.loc[prev_idx, 'balance buy'] - withdrew
                    df.loc[idx, 'balance buy total payed'] = (
                        df.loc[prev_idx, 'balance buy total payed']
                        - withdrew * df.loc[prev_idx, 'avg buy total price']
                    )

            elif info['operation'] == 'sell':
                # sell takes money if available in this order from:
                #  [deposit, stock dividend, dividend, buy]
                # sell is negative, use absolute
                sold = abs(df.loc[idx, 'size'])

                if sold > 0 and df.loc[prev_idx, 'balance deposit'] > 0:
                    if sold > df.loc[prev_idx, 'balance deposit']:
                        sold = sold - df.loc[prev_idx, 'balance deposit']
                        df.loc[idx, 'balance deposit'] = 0
                    else:
                        df.loc[idx, 'balance deposit'] = df.loc[prev_idx, 'balance deposit'] - sold
                        sold = 0
                if sold > 0 and df.loc[prev_idx, 'balance stock dividend'] > 0:
                    if sold > df.loc[prev_idx, 'balance stock dividend']:
                        sold = sold - df.loc[prev_idx, 'balance stock dividend']
                        df.loc[idx, 'balance stock dividend'] = 0
                    else:
                        df.loc[idx, 'balance stock dividend'] = (
                            df.loc[prev_idx, 'balance stock dividend'] - sold
                        )
                        sold = 0
                if sold > 0 and df.loc[prev_idx, 'balance dividend'] > 0:
                    if sold > df.loc[prev_idx, 'balance dividend']:
                        sold = sold - df.loc[prev_idx, 'balance dividend']
                        df.loc[idx, 'balance dividend'] = 0
                    else:
                        df.loc[idx, 'balance dividend'] = (
                            df.loc[prev_idx, 'balance dividend'] - sold
                        )
                        sold = 0
                if sold > 0:
                    df.loc[idx, 'balance buy'] = df.loc[prev_idx, 'balance buy'] - sold
                    df.loc[idx, 'balance buy total payed'] = (
                        df.loc[prev_idx, 'balance buy total payed']
                        - sold * df.loc[prev_idx, 'avg buy total price']
                    )

                    # Compute profit or loss
                    profit_loss = (
                        sold * df.loc[idx, 'price_w_expenses']
                        - sold * df.loc[prev_idx, 'avg buy total price']
                    )
                    df.loc[idx, 'sell profit loss'] = profit_loss

            # Column 'avg buy total price' is set to float('nan') before the for loop
            #  doing `df['avg buy total price'] = float('nan')`
            #  so only those rows where float('nan') should change are set using this
            if df.loc[idx, 'balance buy'] != 0:
                df.loc[idx, 'avg buy total price'] = (
                    df.loc[idx, 'balance buy total payed'] / df.loc[idx, 'balance buy']
                )

            prev_idx = idx  # Keep track of last id for next iteration

        # Compute 'accumulated sell profit loss'
        df['accumulated sell profit loss'] = df['sell profit loss'].cumsum()

        return df

    @_deco_check_ledger_for_cols
    def cols_operation_balance_by_instrument(
        self, show_instr_accnt=False, simplify_dtypes=True
    ) -> pd.DataFrame:
        """Returns a DataFrame with a balance per operation per instrument/account.

        Parameters
        ----------
        show_instr_accnt : bool, optional
            Whether or not to show the instrument and the account. By default False.
        simplify_dtypes : bool, optional
            Allows to simplify dtypes, for instance, pass from float64 to int64 if no decimals are present. Doesn't convert to a dtype that supports pd.NA, like `DataFrame.convert_dtypes()` although it uses it. See https://github.com/pandas-dev/pandas/issues/58543#issuecomment-2101240339 . Warning: Might have a performance impact if True. By default True.

        Returns
        -------
        pd.DataFrame
            A DataFrame with a balance per operation per instrument/account.
        """

        new_columns = [
            'balance deposit',
            'balance dividend',
            'balance stock dividend',
            'balance buy',
            'balance buy total payed',  # Amount used to buy an instrument
            'avg buy total price',
            # Balance for invest in cols_operation_cumsum, called 'invest cumsum'
            # Balance for uninvest in cols_operation_cumsum, called 'uninvest cumsum'
            # Withdraw balance doesn't make sense because it means removing something
            # Sell balance doesn't make sense because it means removing something
            'sell profit loss',
            'accumulated sell profit loss',
        ]

        if len(self._ledger_df) == 0:
            # Return an empty DataFrame with whe structure needed
            to_return = pd.DataFrame(columns=['instrument', 'account', *new_columns])
        else:
            # We only use the columns necessary for the grouping
            cols_subset = [
                'operation',
                'instrument',
                'price_w_expenses',
                'size',
                'stated_total',
                'account',
            ]
            groups = self._ledger_df[cols_subset].groupby(['instrument', 'account'])

            # IMPORTANT:
            # - Columns should not be added manually in this step if the ledger
            #    doesn't include all possible operations, this will be done in
            #    `self._cols_operation_balance_by_instrument_for_group` so
            #    --> DON'T do `.reindex(new_columns, axis=1, fill_value=pd.NA)`
            # - Do not fill na with 0, as this will overwrite the expected
            #    behavior for column 'avg buy total price', which is sometimes nan
            #    when it isn't calculable, so
            #    --> DON'T do `.fillna(0)`
            to_return = (
                groups.apply(
                    self._cols_operation_balance_by_instrument_for_group,
                    include_groups=False,
                    new_columns=new_columns,
                )
                # Move indexes 'instrument' and 'account'created by the last groupby to a column
                .reset_index(['instrument', 'account'])
                # Sort by the original index
                .sort_index()
                # Try to pass colums where dtype is object to a type like int64 or float64
                .infer_objects()
            )

            if simplify_dtypes is True:
                with pd.option_context('future.no_silent_downcasting', True):
                    to_return = (
                        to_return
                        # See https://github.com/pandas-dev/pandas/issues/58543#issuecomment-2101240339
                        .astype('object')
                        .convert_dtypes()
                        .astype('object')
                        .replace(pd.NA, float('nan'))
                        .infer_objects()
                    )

        if show_instr_accnt is True:
            return to_return[['instrument', 'account', *new_columns]]
        else:
            return to_return[[*new_columns]]

    def rm_empty_rows(self):
        """Remove empty rows (where everything is filled with na)."""
        self._ledger_df.drop(
            self._ledger_df[self._ledger_df.isna().all(axis=1)].index,
            inplace=True,
        )

    # *****************
    # MARK: OPERATIONS
    # *****************

    def _add_row(self, value_dict):
        self._ledger_df = (
            pd.concat([self._ledger_df, pd.DataFrame([value_dict])])
            # Make the index coherent (else index might have duplicates)
            .reset_index(drop=True)
        )

    def buy(self):
        pass

    def deposit(
        self,
        date_execution,
        instrument,
        amount,
        account,
        date_order=None,
        description='',
        notes='',
    ):
        """Creates a new row with a deposit.

        Important: Deposit means a instrument entering an account. A commission or tax for a deposit is not possible, create a new operation for that purpose after doing the deposit.

        Parameters
        ----------
        date_execution : _type_
            Deposit execution date.
        instrument : _type_
            Deposit instrument.
        amount : _type_
            Deposit amount.
        account : _type_
            Deposit account.
        date_order : _type_, optional
            Deposit order date, if not set, uses `date_execution`. By default None.
        description : str, optional
            Deposit description. By default ''.
        notes : str, optional
            Deposit notes. By default ''.
        """
        self._add_row(
            {
                'date_execution': date_execution,
                'operation': 'deposit',
                'instrument': instrument,
                'origin': '',
                'destination': instrument,
                'price_in': instrument,
                'price': 1,
                'price_w_expenses': 1,
                'size': amount,
                'commission': 0,
                'tax': 0,
                'stated_total': amount,
                'date_order': (date_order if date_order is not None else date_execution),
                'description': description,
                'notes': notes,
                'commission_notes': '',
                'tax_notes': '',
                'account': account,
                'Q_price_commission_tax_verification': 0,
            }
        )

    def dividend(self):
        pass

    def invest(self):
        pass

    def sell(self):
        pass

    def stock_dividend(self):
        pass

    def uninvest(self):
        pass

    def withdraw(
        self,
        date_execution,
        instrument,
        amount,
        account,
        date_order=None,
        description='',
        notes='',
    ):
        """Creates a new row with a withdraw.

        Important: Withdraw means a instrument exiting an account. A commission or tax for a withdraw is not possible, create a new operation for that purpose after doing the withdraw.

        Parameters
        ----------
        date_execution : _type_
            Withdraw execution date.
        instrument : _type_
            Withdraw instrument.
        amount : _type_
            Withdraw amount.
        account : _type_
            Withdraw account.
        date_order : _type_, optional
            Withdraw order date, if not set, uses `date_execution`. By default None.
        description : str, optional
            Withdraw description. By default ''.
        notes : str, optional
            Withdraw notes. By default ''.
        """
        self._add_row(
            {
                'date_execution': date_execution,
                'operation': 'withdraw',
                'instrument': instrument,
                'origin': instrument,
                'destination': '',
                'price_in': instrument,
                'price': 1,
                'price_w_expenses': 1,
                'size': -amount,
                'commission': 0,
                'tax': 0,
                'stated_total': -amount,
                'date_order': (date_order if date_order is not None else date_execution),
                'description': description,
                'notes': notes,
                'commission_notes': '',
                'tax_notes': '',
                'account': account,
                'Q_price_commission_tax_verification': 0,
            }
        )

    # *****************
    # MARK: METADATA
    # *****************

    def get_present_instruments(self) -> np.ndarray:
        """Returns present instruments in the ledger.

        Returns
        -------
        np.ndarray
            Present instruments in the ledger.
        """
        return self._ledger_df['instrument'].unique()

    def get_present_operations(self) -> np.ndarray:
        """Returns present operations in the ledger.

        Returns
        -------
        np.ndarray
            Present operations in the ledger.
        """
        return self._ledger_df['operation'].unique()

    def get_instruments_metadata(self) -> dict:
        """Returns a dictionary of instruments' metadata.

        The returned format is the following:
        ```json
        {instrument_name:
            {'type': instrument_type,
            'name': instrument_name}}
        ```

        For example:
        ```json
        {
            'AAPL': {'name': 'Apple Inc.', 'type': 'stock'},
            'USD': {'name': 'USA Dollar', 'type': 'cash'},
        }
        ```

        Returns
        -------
        dict
            Dictionary of instruments' metadata.
        """
        return self._instruments_metadata

    def set_instruments_metadata(self, instuments_metadata: dict) -> None:
        """Set instruments' metadata.

        Parameters
        ----------
        instuments_metadata : dict
            The required format is the following:👍🏼
            ```json
            {instrument_name:
                {'type': instrument_type,
                'name': instrument_name}}
            ```

            For example:
            ```json
            {
                'AAPL': {'name': 'Apple Inc.', 'type': 'stock'},
                'USD': {'name': 'USA Dollar', 'type': 'cash'},
            }
            ```
        """
        extraneous_instruments = set(instuments_metadata.keys()) - set(
            self.get_present_instruments()
        )
        if len(extraneous_instruments) > 0:
            warnings.warn(
                'Some instruments provided in the dict are not present in the ledger: '
                + f'{str(extraneous_instruments)}'
                + ' (still, storing all in the instruments metadata).',
                stacklevel=2,
            )
        for instr, metadata in instuments_metadata.items():
            if not isinstance(metadata, dict):
                continue
            if metadata.get('name') is not None:  # Does not override
                namedict = {'name': metadata['name']}
            if metadata.get('type') is not None:  # Does not override
                typedict = {'type': metadata['type']}
            self._instruments_metadata[instr] = {**namedict, **typedict}

    def rm_instruments_metadata(self, instruments: str | List[str]) -> None:
        """Remove instrument(s) from the metadata.

        Args:
            instruments (str | List[str]):
                If a string, it removes the instrument from the metadata.

                If a list, it removes each instrument in the list from the metadata.
        """
        if isinstance(instruments, str):
            self._instruments_metadata.pop(instruments, None)
        if isinstance(instruments, list):
            for instr in instruments:
                if isinstance(instr, str):
                    self._instruments_metadata.pop(instr, None)

    def instruments(
        self,
        instrument_type: bool = True,
        instrument_name: bool = True,
        instrument_in_ledger: bool = True,
    ) -> pd.DataFrame:
        """Returns instruments present in the ledger and in the instruments' metadata.

        The instrument present in the ledger are those that take part in a transaction in the column 'instrument'.

        Some instruments might be present in the instruments' metadata and not in the ledger, those are also returned.

        Parameters
        ----------
        instrument_type : bool, optional
            Whether to return the types in an 'instrument_type' column. By default True.
        instrument_name : bool, optional
            Whether to return the names in an 'instrument_name' column. By default True.
        instrument_in_ledger : bool, optional
            Whether to return a column 'in_ledger' specifying if an instrument is in the ledger or not. By default True.

        Returns
        -------
        pd.DataFrame
            _description_
        """
        instruments_set = set([*self.get_present_instruments(), *self._instruments_metadata.keys()])

        dfs_toconcat = [pd.DataFrame(instruments_set, columns=['instrument'])]

        if instrument_type is True:
            rows_types = []
            for instr in instruments_set:
                metadata = self._instruments_metadata.get(instr)
                instr_type = metadata.get('type', '') if isinstance(metadata, dict) else ''
                rows_types.append([instr_type])
            dfs_toconcat.append(pd.DataFrame(rows_types, columns=['instrument_type']))

        if instrument_name is True:
            rows_names = []
            for instr in instruments_set:
                metadata = self._instruments_metadata.get(instr)
                instr_name = metadata.get('name', '') if isinstance(metadata, dict) else ''
                rows_names.append([instr_name])
            dfs_toconcat.append(pd.DataFrame(rows_names, columns=['instrument_name']))

        if instrument_in_ledger is True:
            extraneous_instruments = set(self._instruments_metadata.keys()) - set(
                self.get_present_instruments()
            )
            rows_extraneous = []
            for instr in instruments_set:
                is_extraneous = False if instr in extraneous_instruments else True
                rows_extraneous.append([is_extraneous])
            dfs_toconcat.append(pd.DataFrame(rows_extraneous, columns=['in_ledger']))

        return (
            pd
            # Join DataFrames
            .concat(dfs_toconcat, join='inner', axis=1)
            # Try to pass colums where dtype is object to a type like int64 or float64
            .infer_objects()
            .sort_values('instrument')
            .reset_index(drop=True)
        )
