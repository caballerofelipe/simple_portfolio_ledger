'''A simple ledger to keep track of a Portfolio's movements.'''

import datetime
import functools
import inspect
import pathlib
import warnings
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, validate_call

# Note: MARK is a functionality from Visual Studio Code to find things in the minimap.


class SimplePortfolioLedger:

    _ledger_columns = (
        'opid',
        'subopid',
        'account',
        'date_execution',
        'operation',
        'instrument',
        'size',
        'price_in',
        'origin',
        'destination',
        'price',
        'price_w_expenses',
        'commission',
        'tax',
        'total',
        'date_order',
        'description',
        'notes',
        'commission_notes',
        'tax_notes',
    )

    # Column description in a dict
    _ledger_columns_attrs = {
        'column notes': {
            'opid': 'Operation id, used to keep track of multiple rows operations.',
            'subopid': 'Sub operation id, used to keep track of sub operations.',
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
        }
    }

    _ops_names = set(
        (
            'buy',
            'deposit',
            'dividend',
            'invest',
            'sell',
            'stock_dividend',
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
    def _deco_ledger_check(func: Callable) -> Callable:
        """This decorator checks the ledger before invoking it or some computation function.

        It does the following:
        - If the ledger is empty, it issues a warning stating that the function will only return the basic structure.
        - Checks if a foreign operation was added to the ledger, in which case an exception is raised.

        Parameters
        ----------
        func : Callable
            The function that will show the ledger or a computation of the ledger.

        Returns
        -------
        Callable
            The wrapper function.

        Raises
        ------
        ValueError
            'One or more forbidden operations were inserted in the ledger.'
        """

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
            warnings.warn('Columns must be list or str. Returning None.', stacklevel=2)
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

    # *****************
    # MARK: Computing Operations
    # *****************

    def _get_extraneous_ops(self):
        # Compute extraneous_ops, which shows all unique operations from self._ledger_df
        #  minus all operations self._ops_names
        #  leaving operations that shouldn't be here
        used_ops = set(self._ledger_df['operation'].unique())
        extraneous_ops = used_ops - set(self._ops_names)
        return extraneous_ops

    @staticmethod
    @validate_call(config={'arbitrary_types_allowed': True})
    def simplify_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Allows to simplify dtypes, for instance, pass from float64 to int64 if no decimals are present.

        Doesn't convert to a dtype that supports pd.NA, like `DataFrame.convert_dtypes()` although it uses it. See https://github.com/pandas-dev/pandas/issues/58543#issuecomment-2101240339 . It might create a performance impact but this hasn't been tested.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to simplify.

        Returns
        -------
        pd.DataFrame
           The DataFrame, with simplified dtypes.
        """
        with pd.option_context('future.no_silent_downcasting', True):
            return (
                df
                # See https://github.com/pandas-dev/pandas/issues/58543#issuecomment-2101240339
                .astype('object')
                .convert_dtypes()
                .astype('object')
                .replace(pd.NA, float('nan'))
                .infer_objects()
            )

    @_deco_ledger_check
    def ledger(
        self,
        instrument_type: bool = False,
        instrument_name: bool = False,
        operation_columns: bool = False,
        instrument_cumsum_by_operation: bool = False,
        operation_columns_balance: bool = False,
        thousands_fmt_sep=False,
        thousands_fmt_decimals=1,
    ) -> pd.DataFrame:
        """Returns The Ledger, with optional additional information.

        Parameters
        ----------
        instrument_type : bool, optional
            Wether to add a column for the instrument type. By default False.
        instrument_name : bool, optional
            Wether to add a column for the instrument name. By default False.
        operation_columns : bool, optional
            Whether to add operation_columns to The Ledger. By default False.
        instrument_cumsum_by_operation : bool, optional
            Whether to add instrument_cumsum_by_operation to The Ledger. By default False.
        operation_columns_balance : bool, optional
            Whether to add operation_columns_balance to The Ledger. By default False.
        thousands_fmt_sep : bool, optional
            Add a thousands separator. By default False.
        thousands_fmt_decimals : int, optional
            Decimals to print, used only when thousands_fmt_sep is set to True. By default 1.

        Returns
        -------
        pd.DataFrame
            The Ledger, with optional additional information.
        """

        # Simplify dtypes
        the_ledger = self.simplify_dtypes(
            # Specifying columns to avoid returning manually added columns
            # `*` needed because self._ledger_columns is a tuple
            self._ledger_df[[*self._ledger_columns]],
        )

        if instrument_type is True or instrument_name is True:
            instruments = self.instruments(
                instrument_type=instrument_type,
                instrument_name=instrument_name,
                instrument_in_ledger=False,
            )
            the_ledger = pd.merge(the_ledger, instruments, how='left', on='instrument')

        dfs_to_concat = [the_ledger]

        if operation_columns is True:
            tmp = self.operation_columns()
            dfs_to_concat.append(tmp)
        if instrument_cumsum_by_operation is True:
            tmp = self.instrument_cumsum_by_operation(all_columns=False)
            dfs_to_concat.append(tmp)
        if operation_columns_balance is True:
            tmp = self.operation_columns_balance(all_columns=False)
            dfs_to_concat.append(tmp)

        to_return = (
            pd
            # Join DataFrames
            .concat(dfs_to_concat, join='inner', axis=1)
            # Convert dates to appropriate format
            .astype(
                {
                    'date_execution': 'datetime64[ns]',
                    'date_order': 'datetime64[ns]',
                }
            )
            # Try to pass columns where dtype is object to a type like int64 or float64
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

    @_deco_ledger_check
    def operation_columns(
        self,
    ) -> pd.DataFrame:
        """Returns a DataFrame with each operation as a column.

        Returns
        -------
        pd.DataFrame
            DataFrame with each operation as a column.
        """

        if len(self._ledger_df) == 0:
            # Return an empty DataFrame with whe structure needed
            to_return = pd.DataFrame(columns=[*sorted(self._ops_names)])
        else:
            # Create groups by instrument/operation
            groups = self._ledger_df.groupby(['operation'])

            # For each group, return the size
            #  (for example if an op is buy, the column buy would be filled, not all others)
            op_size = groups.apply(lambda x: x['size'], include_groups=False)

            with pd.option_context('future.no_silent_downcasting', True):
                to_return = (
                    op_size
                    # Move operation from the index and create one column for each op
                    .unstack('operation')
                    # Remove name for columns' index
                    .rename_axis(None, axis=1)
                    # Fill na with 0
                    .fillna(0)
                    # Add columns that weren't created in the previous operation
                    #   using the operation list from the class self._ops_names
                    .reindex(sorted(self._ops_names), axis=1, fill_value=0)
                    # Sort by the original index
                    .sort_index()
                    # Try to pass columns where dtype is object to a type like int64 or float64
                    .infer_objects()
                )

                to_return = self.simplify_dtypes(to_return)

        return to_return[[*sorted(self._ops_names)]]

    @_deco_ledger_check
    def instrument_cumsum_by_operation(
        self,
        all_columns=True,
        group_by_account=False,
        group_by_price_in=False,
    ) -> pd.DataFrame:
        """Returns a DataFrame with a cumsum for each operation for each account-instrument-price_in.

        Useful to see the total "so far" for each account-instrument-price_in combination.

        The index of the returned DataFrame is the same as the ledger but might be sorted differently.

        The operation is done by grouping operations by account-instrument-price_in, doing a cumsum for every group and returns a DataFrame with the following columns:
            - 'account' (when all_columns=True)
            - 'instrument' (when all_columns=True)
            - 'price_in' (when all_columns=True)
            - The cumsum of all operation, every operation turned into a column.
            - 'cumsum_held'
            - 'cumsum_invested'

        Parameters
        ----------
        all_columns : bool, optional
            Whether or not to show columns 'account', 'instrument', 'price_in'. By default True.

        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns:
                - 'account' (when all_columns=True)
                - 'instrument' (when all_columns=True)
                - 'price_in' (when all_columns=True)
                - The cumsum of all operation, every operation turned into a column.
                - 'cumsum_held'
                - 'cumsum_invested'
        """

        # List of columns to return, SORTED by self._ops_names
        ops_cumsum_names = [f'cumsum_{c}' for c in sorted(self._ops_names)]

        returned_columns = [
            'account',  # Could be removed next
            'instrument',  # MUST NOT be removed
            'price_in',  # Could be removed next
        ]

        # Note: instead of adding, this code removes items
        # as we want to have a pre specified order stated above
        if group_by_account is False:
            returned_columns.pop(returned_columns.index('account'))
        if group_by_price_in is False:
            returned_columns.pop(returned_columns.index('price_in'))

        if len(self._ledger_df) == 0:
            # Return an empty DataFrame with whe structure needed
            to_return = pd.DataFrame(
                columns=[
                    *returned_columns,
                    *ops_cumsum_names,
                    'cumsum_held',
                    'cumsum_invested',
                ]
            )
        else:
            # Create groups by instrument/operation
            groups = self._ledger_df.groupby([*returned_columns, 'operation'])

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
                    # Remove name for columns' index
                    .rename_axis(None, axis=1)
                    # Add columns that weren't created in the previous operation
                    #  using the operation list from the class' self._ops_names
                    .reindex(sorted(self._ops_names), axis=1, fill_value=0)
                    # Rename created columns
                    .rename(columns=rename_dict)
                    # Move indexes `returned_columns` (created by the last groupby) to a column
                    .reset_index(returned_columns)
                    # Regroup to do the ffill
                    .groupby(returned_columns)
                    # ffill with the cumsum, fill with 0 the beginning
                    .apply(lambda x: x.ffill().fillna(0), include_groups=False)
                    # Move indexes `returned_columns` (created by the last groupby) to a column
                    .reset_index(returned_columns)
                    # Create new columns
                    .assign(
                        # sum of all cumsum columns
                        cumsum_held=lambda x: x[ops_cumsum_names].sum(axis=1),
                        # Invested total in positive value
                        cumsum_invested=(
                            lambda x: x[['cumsum_invest', 'cumsum_uninvest']].sum(axis=1).abs()
                        ),
                    )
                    # Sort by the original index
                    .sort_index()
                    # Try to pass columns where dtype is object to a type like int64 or float64
                    .infer_objects()
                )

                to_return = self.simplify_dtypes(to_return)

        if all_columns is True:
            return to_return[
                [
                    *returned_columns,
                    *ops_cumsum_names,
                    'cumsum_held',
                    'cumsum_invested',
                ]
            ]
        else:
            return to_return[[*ops_cumsum_names, 'cumsum_held', 'cumsum_invested']]

    @staticmethod
    def _operation_columns_balance_for_group(group_df, new_columns) -> pd.DataFrame:
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
        df['avg_buy_total_price'] = float('nan')

        # Cols to pass from previous row to current
        cols_to_copy = [col for col in new_columns if 'balance' in col]

        # Initially prev index is the same as current index
        prev_idx = df.index[0]

        for row in df.iterrows():
            idx, info = row

            # Copy previous operation (index: prev_idx) balance
            #  (in this loop the current operation (index: idx) will be changed)
            #  This is necessary as some operations don't touch every new column
            #  and this assures the untouched operations keep track of history
            df.loc[idx, cols_to_copy] = df.loc[prev_idx, cols_to_copy]

            # invest and uninvest should not be part of the balance
            #  the invested balance is computed in `instrument_cumsum_by_operation`, this applies to:
            # `info['operation'] == 'invest'` and `info['operation'] == 'uninvest'`

            if info['operation'] == 'deposit':
                df.loc[idx, 'balance_deposit'] = (
                    df.loc[prev_idx, 'balance_deposit'] + df.loc[idx, 'size']
                )

            elif info['operation'] == 'buy':
                df.loc[idx, 'balance_buy'] = df.loc[prev_idx, 'balance_buy'] + df.loc[idx, 'size']
                df.loc[idx, 'balance_buy_total_payed'] = (
                    df.loc[prev_idx, 'balance_buy_total_payed'] + df.loc[idx, 'total']
                )
                # df.loc[idx, 'avg_buy_total_price'] will be computed at the end
                #  of each iteration of this for

            elif info['operation'] == 'dividend':
                df.loc[idx, 'balance_dividend'] = (
                    df.loc[prev_idx, 'balance_dividend'] + df.loc[idx, 'size']
                )

            elif info['operation'] == 'stock_dividend':
                df.loc[idx, 'balance_stock_dividend'] = (
                    df.loc[prev_idx, 'balance_stock_dividend'] + df.loc[idx, 'size']
                )

            elif info['operation'] == 'withdraw':
                # withdraw takes money if available in this order from:
                #  [deposit, stock dividend, dividend, buy]
                # withdraw is negative, use absolute
                withdrew = abs(df.loc[idx, 'size'])

                if withdrew > 0 and df.loc[prev_idx, 'balance_deposit'] > 0:
                    if withdrew > df.loc[prev_idx, 'balance_deposit']:
                        withdrew = withdrew - df.loc[prev_idx, 'balance_deposit']
                        df.loc[idx, 'balance_deposit'] = 0
                    else:
                        df.loc[idx, 'balance_deposit'] = (
                            df.loc[prev_idx, 'balance_deposit'] - withdrew
                        )
                        withdrew = 0
                if withdrew > 0 and df.loc[prev_idx, 'balance_stock_dividend'] > 0:
                    if withdrew > df.loc[prev_idx, 'balance_stock_dividend']:
                        withdrew = withdrew - df.loc[prev_idx, 'balance_stock_dividend']
                        df.loc[idx, 'balance_stock_dividend'] = 0
                    else:
                        df.loc[idx, 'balance_stock_dividend'] = (
                            df.loc[prev_idx, 'balance_stock_dividend'] - withdrew
                        )
                        withdrew = 0
                if withdrew > 0 and df.loc[prev_idx, 'balance_dividend'] > 0:
                    if withdrew > df.loc[prev_idx, 'balance_dividend']:
                        withdrew = withdrew - df.loc[prev_idx, 'balance_dividend']
                        df.loc[idx, 'balance_dividend'] = 0
                    else:
                        df.loc[idx, 'balance_dividend'] = (
                            df.loc[prev_idx, 'balance_dividend'] - withdrew
                        )
                        withdrew = 0
                if withdrew > 0:
                    df.loc[idx, 'balance_buy'] = df.loc[prev_idx, 'balance_buy'] - withdrew
                    df.loc[idx, 'balance_buy_total_payed'] = (
                        df.loc[prev_idx, 'balance_buy_total_payed']
                        - withdrew * df.loc[prev_idx, 'avg_buy_total_price']
                    )

            elif info['operation'] == 'sell':
                # sell takes money if available in this order from:
                #  [deposit, stock dividend, dividend, buy]
                # sell is negative, use absolute
                sold = abs(df.loc[idx, 'size'])

                if sold > 0 and df.loc[prev_idx, 'balance_deposit'] > 0:
                    if sold > df.loc[prev_idx, 'balance_deposit']:
                        sold = sold - df.loc[prev_idx, 'balance_deposit']
                        df.loc[idx, 'balance_deposit'] = 0
                    else:
                        df.loc[idx, 'balance_deposit'] = df.loc[prev_idx, 'balance_deposit'] - sold
                        sold = 0
                if sold > 0 and df.loc[prev_idx, 'balance_stock_dividend'] > 0:
                    if sold > df.loc[prev_idx, 'balance_stock_dividend']:
                        sold = sold - df.loc[prev_idx, 'balance_stock_dividend']
                        df.loc[idx, 'balance_stock_dividend'] = 0
                    else:
                        df.loc[idx, 'balance_stock_dividend'] = (
                            df.loc[prev_idx, 'balance_stock_dividend'] - sold
                        )
                        sold = 0
                if sold > 0 and df.loc[prev_idx, 'balance_dividend'] > 0:
                    if sold > df.loc[prev_idx, 'balance_dividend']:
                        sold = sold - df.loc[prev_idx, 'balance_dividend']
                        df.loc[idx, 'balance_dividend'] = 0
                    else:
                        df.loc[idx, 'balance_dividend'] = (
                            df.loc[prev_idx, 'balance_dividend'] - sold
                        )
                        sold = 0
                if sold > 0:
                    df.loc[idx, 'balance_buy'] = df.loc[prev_idx, 'balance_buy'] - sold
                    df.loc[idx, 'balance_buy_total_payed'] = (
                        df.loc[prev_idx, 'balance_buy_total_payed']
                        - sold * df.loc[prev_idx, 'avg_buy_total_price']
                    )

                    # Compute profit or loss
                    profit_loss = (
                        sold * df.loc[idx, 'price_w_expenses']
                        - sold * df.loc[prev_idx, 'avg_buy_total_price']
                    )
                    df.loc[idx, 'sell_profit_loss'] = profit_loss

            # Column 'avg_buy_total_price' is set to float('nan') before the for loop
            #  doing `df['avg_buy_total_price'] = float('nan')`
            #  so only those rows where float('nan') should change are set using this
            if df.loc[idx, 'balance_buy'] != 0:
                df.loc[idx, 'avg_buy_total_price'] = (
                    df.loc[idx, 'balance_buy_total_payed'] / df.loc[idx, 'balance_buy']
                )

            prev_idx = idx  # Keep track of last id for next iteration

        # Compute 'accumulated_sell_profit_loss'
        df['accumulated_sell_profit_loss'] = df['sell_profit_loss'].cumsum()

        return df

    @_deco_ledger_check
    def operation_columns_balance(self, all_columns=True) -> pd.DataFrame:
        """Returns a DataFrame with a balance per operation per account-instrument-price_in.

        Useful to see the balance on every row for each account-instrument-price_in combination.

        The index of the returned DataFrame is the same as the ledger but might be sorted differently.

        The operation is done by grouping operations by account-instrument-price_in, doing a balance for every group and returns a DataFrame with the following columns:
            - 'account' (when all_columns=True)
            - 'instrument' (when all_columns=True)
            - 'price_in' (when all_columns=True)
            - 'balance_deposit',
            - 'balance_dividend',
            - 'balance_stock_dividend',
            - 'balance_buy',
            - 'balance_buy_total_payed',  # Amount used to buy an instrument
            - 'avg_buy_total_price',
            - 'sell_profit_loss',
            - 'accumulated_sell_profit_loss'

        From all the operations, the following aren't computed, and an explanation is given:
            - 'invest', balance for invest is in instrument_cumsum_by_operation, called 'invest cumsum'.
            - 'uninvest', balance for uninvest is in instrument_cumsum_by_operation, called 'uninvest cumsum'.
            - 'withdraw' balance doesn't make sense because it means removing something, it is incorporated into one of 'balance_deposit', 'balance_stock_dividend', 'balance_dividend', 'balance_buy'.
            - 'sell' balance doesn't make sense because it means removing something, it is incorporated into one of 'balance_deposit', 'balance_stock_dividend', 'balance_dividend', 'balance_buy'.

        Parameters
        ----------
        all_columns : bool, optional
            Whether or not to show columns 'account', 'instrument', 'price_in'. By default True.

        Returns
        -------
        pd.DataFrame
            A DataFrame with a balance per operation per instrument/account.
        """

        new_columns = [
            'balance_deposit',
            'balance_dividend',
            'balance_stock_dividend',
            'balance_buy',
            'balance_buy_total_payed',  # Amount used to buy an instrument
            'avg_buy_total_price',
            'sell_profit_loss',
            'accumulated_sell_profit_loss',
            # 'invest', balance for invest is in instrument_cumsum_by_operation, called 'invest cumsum'
            # 'uninvest', balance for uninvest is in instrument_cumsum_by_operation, called 'uninvest cumsum'
            # 'withdraw' balance doesn't make sense because it means removing something
            # 'sell' balance doesn't make sense because it means removing something
        ]

        if len(self._ledger_df) == 0:
            # Return an empty DataFrame with whe structure needed
            to_return = pd.DataFrame(columns=['account', 'instrument', 'price_in', *new_columns])
        else:
            # We only use the columns necessary for the grouping
            cols_subset = [
                'account',
                'instrument',
                'price_in',
                'operation',
                'price_w_expenses',
                'size',
                'total',
            ]
            groups = self._ledger_df[cols_subset].groupby(['account', 'instrument', 'price_in'])

            # IMPORTANT:
            # - Columns should not be added manually in this step if the ledger
            #    doesn't include all possible operations, this will be done in
            #    `self._operation_columns_balance_for_group` so
            #    --> DON'T do `.reindex(new_columns, axis=1, fill_value=pd.NA)`
            # - Do not fill na with 0, as this will overwrite the expected
            #    behavior for column 'avg_buy_total_price', which is sometimes nan
            #    when it isn't calculable, so
            #    --> DON'T do `.fillna(0)`
            to_return = (
                groups.apply(
                    self._operation_columns_balance_for_group,
                    include_groups=False,
                    new_columns=new_columns,
                )
                # Move indexes 'instrument' and 'account'created by the last groupby to a column
                .reset_index(['account', 'instrument', 'price_in'])
                # Sort by the original index
                .sort_index()
                # Try to pass columns where dtype is object to a type like int64 or float64
                .infer_objects()
            )

            to_return = self.simplify_dtypes(to_return)

        if all_columns is True:
            return to_return[['account', 'instrument', 'price_in', *new_columns]]
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

    def _add_rows(self, data: dict | List[dict]) -> tuple[int, pd.DataFrame]:
        """Add one or multiple rows to the ledger.

        The new row(s) will have an opid incremented by one from the max opid in the ledger. Or it will be 0 if max is a nan.

        Parameters
        ----------
        data : dict | List[dict]
            The data to be added to the ledger. If a dict, the function adds only one row. If a list of dicts, the function adds as many items in the list.

        Returns
        -------
        tuple[int, pd.DataFrame]
            Returns a tuple of the inserted opid and the pd.DataFrame inserted.

        Raises
        ------
        ValueError
            If data is a list and an item in that list is not a dict, an error is raised.
        ValueError
            If data is neither a dict nor a List, an error is raised.
        """
        row_list = []

        # Get the last opid and set the one used for the next row
        last_opid = self._ledger_df['opid'].max()
        if pd.isna(last_opid):
            new_op_id = 0
        else:
            new_op_id = int(last_opid) + 1

        # Fill the array of rows to be added
        if isinstance(data, dict):
            row_list.append({**data, 'opid': new_op_id, 'subopid': 0})
        elif isinstance(data, list):
            for i, vd in enumerate(data):
                if isinstance(vd, dict):
                    row_list.append({**vd, 'opid': new_op_id, 'subopid': i})
                else:
                    raise ValueError('When creating a row, the item in the list is not a dict.')
        else:
            raise ValueError('Unsupported type for value_dict, must be a dict or a list of dicts.')

        # Add rows to the ledger
        # Specifying columns order, `*` needed because self._ledger_columns is a tuple
        new_rows = pd.DataFrame(row_list)[[*self._ledger_columns]]
        self._ledger_df = (
            pd.concat([self._ledger_df, new_rows])
            # Make the index coherent (else index might have duplicates)
            .reset_index(drop=True)
        )
        return new_op_id, new_rows

    @validate_call(config={'arbitrary_types_allowed': True})
    def buy(
        self,
        account: str,
        date_execution: datetime.datetime,
        instrument: str,
        size: PositiveInt | PositiveFloat,
        price_in: str,
        price: NonNegativeInt | NonNegativeFloat,
        commission: NonNegativeInt | NonNegativeFloat = 0,
        tax: NonNegativeInt | NonNegativeFloat = 0,
        date_order: datetime.datetime | None = None,
        notes: str = '',
        commission_notes: str = '',
        tax_notes: str = '',
        stated_total: NonNegativeInt | NonNegativeFloat | None = None,
        tolerance_decimals: NonNegativeInt = 4,
    ):
        """Creates the operations needed to process a buy.

        Parameters
        ----------
        account : str
            Account used for the operations.
        date_execution : str
            Date when the operations are done.
        instrument : str
            The instrument that increases its amount by the operations.
        size : PositiveInt | PositiveFloat
            The amount bought.
        price_in : str
            The instrument used to pay for the buy.
        price : NonNegativeInt | NonNegativeFloat
            The current market price for the instrument.
        commission : NonNegativeInt | NonNegativeFloat, optional
            The amount payed to do the operation in `price_in`, by default 0.
        tax : NonNegativeInt | NonNegativeFloat, optional
            The amount payed in taxes in `price_in`, by default 0.
        date_order : str | None, optional
            Date when the operations are ordered. By default None, in which case `date_execution` is used.
        notes : str, optional
            Notes to add to the operations, by default ''.
        commission_notes : str, optional
            Notes about the commission to add to the operations, by default ''.
        tax_notes : str, optional
            Notes about the tax to add to the operations, by default ''.
        stated_total : NonNegativeInt  |  NonNegativeFloat  |  None, optional
            Used to verify if the inner calculation (`calculated_total = size * price + commission + tax`) is correct (`abs(stated_total - calculated_total) < 10**-tolerance_decimals`), By default None.

        Returns
        -------
        tuple[int, pd.DataFrame]
            Returns a tuple of the inserted opid and the pd.DataFrame inserted.

        Raises
        ------
        ValueError
            When adding buy operation, the stated total is different from the calculated total. By more than the minimum of {10**-tolerance_decimals}.
        """
        calculated_total = size * price + commission + tax

        # Allow inconsistencies up to 4 decimals
        if stated_total is not None and not (
            abs(stated_total - calculated_total) < 10**-tolerance_decimals
        ):
            error_msg = ''.join(
                [
                    'When adding buy operation, '
                    + 'the stated total is different from the calculated total. '
                    + f'By more than the minimum of {10**-tolerance_decimals}.'
                ]
            )
            error_dict = {
                'date_execution': date_execution,
                'instrument': instrument,
                'stated_total': stated_total,
                'size': size,
                'price': price,
                'commission': commission,
                'tax': tax,
                'calculated_total': calculated_total,
                'calculated_total-stated_total': calculated_total - stated_total,
            }
            raise ValueError(error_msg, error_dict)

        price_w_expenses = calculated_total / size

        op_1 = {}  # invest
        op_2 = {}  # buy
        op_1['date_execution'] = date_execution  # invest
        op_2['date_execution'] = date_execution  # buy
        op_1['operation'] = 'invest'  # invest
        op_2['operation'] = 'buy'  # buy
        op_1['instrument'] = price_in  # invest
        op_2['instrument'] = instrument  # buy
        op_1['origin'] = ''  # invest
        op_2['origin'] = price_in  # buy
        op_1['destination'] = instrument  # invest
        op_2['destination'] = ''  # buy
        op_1['price_in'] = price_in  # invest
        op_2['price_in'] = price_in  # buy
        op_1['price'] = 1  # invest
        op_2['price'] = price  # buy
        op_1['price_w_expenses'] = 1  # invest
        op_2['price_w_expenses'] = price_w_expenses
        op_1['size'] = -1 * calculated_total  # invest
        op_2['size'] = size  # buy
        op_1['commission'] = 0  # invest
        op_2['commission'] = commission  # buy
        op_1['tax'] = 0  # invest
        op_2['tax'] = tax  # buy
        op_1['total'] = -1 * calculated_total  # invest
        op_2['total'] = calculated_total  # buy
        op_1['date_order'] = date_order if date_order is not None else date_execution  # invest
        op_2['date_order'] = date_execution  # buy
        op_1['description'] = f'Invest in {instrument}.'  # invest
        op_2['description'] = f'Buy {instrument}.'  # buy
        op_1['notes'] = notes  # invest
        op_2['notes'] = notes  # buy
        op_1['commission_notes'] = commission_notes  # invest
        op_2['commission_notes'] = commission_notes  # buy
        op_1['tax_notes'] = tax_notes  # invest
        op_2['tax_notes'] = tax_notes  # buy
        op_1['account'] = account  # invest
        op_2['account'] = account  # buy

        return self._add_rows(data=[op_1, op_2])

    @validate_call(config={'arbitrary_types_allowed': True})
    def deposit(
        self,
        account: str,
        date_execution: datetime.datetime,
        instrument: str,
        size: PositiveInt | PositiveFloat,
        date_order: datetime.datetime | None = None,
        notes: str = '',
    ):
        """Creates the operations needed to process a deposit.

        Important: Deposit means a instrument entering an account. A commission or tax for a deposit is not possible, create a new operation for that purpose after doing the deposit.

        Parameters
        ----------
        account : str
            Account used for the operations.
        date_execution : str
            Date when the operations are done.
        instrument : str
            The instrument that increases its size by the operations.
        size : PositiveInt | PositiveFloat
            The amount deposited.
        date_order : str | None, optional
            Date when the operations are ordered. By default None, in which case `date_execution` is used.
        notes : str, optional
            Notes to add to the operations, by default ''.

        Returns
        -------
        tuple[int, pd.DataFrame]
            Returns a tuple of the inserted opid and the pd.DataFrame inserted.
        """
        op_deposit = {
            'date_execution': date_execution,
            'operation': 'deposit',
            'instrument': instrument,
            'origin': '',
            'destination': instrument,
            'price_in': instrument,
            'price': 1,
            'price_w_expenses': 1,
            'size': size,
            'commission': 0,
            'tax': 0,
            'total': size,
            'date_order': (date_order if date_order is not None else date_execution),
            'description': f'Deposit {instrument}.',
            'notes': notes,
            'commission_notes': '',
            'tax_notes': '',
            'account': account,
        }

        return self._add_rows(op_deposit)

    @validate_call(config={'arbitrary_types_allowed': True})
    def dividend(
        self,
        account: str,
        date_execution: datetime.datetime,
        instrument: str,
        total: PositiveInt | PositiveFloat,
        origin: str,
        date_order: datetime.datetime | None = None,
        notes: str = '',
    ):
        """Creates the operations needed to process a dividend.

        Parameters
        ----------
        account : str
            Account used for the operations.
        date_execution : str
            Date when the operations are done.
        instrument : str
            The instrument that increases its amount by the operations.
        total : PositiveInt | PositiveFloat
            The total amount obtained by dividend.
        origin : str
            The instrument that originated the dividend.
        date_order : str | None, optional
            Date when the operations are ordered. By default None, in which case `date_execution` is used.
        notes : str, optional
            Notes to add to the operations, by default ''.

        Returns
        -------
        tuple[int, pd.DataFrame]
            Returns a tuple of the inserted opid and the pd.DataFrame inserted.

        Raises
        ------
        ValueError
            'When doing operation "dividend", the calculated size of held instrument {origin} is 0. This could mean there are no previous operations for the instrument or the sum of all operations is 0. Check the ledger.'
        """
        size = self._ledger_df.query(
            f'instrument == "{origin}" and date_execution <= "{date_execution}"'
        )['size'].sum()

        if size == 0:
            raise ValueError(
                ' '.join(
                    [
                        'When doing operation "dividend",',
                        f'the calculated size of held instrument {origin} is 0.',
                        'This could mean there are no previous operations for the instrument',
                        'or the sum of all operations is 0. Check the ledger.',
                    ]
                )
            )

        price = total / size
        op_dividend = {
            'date_execution': date_execution,
            'operation': 'dividend',
            'instrument': instrument,
            'origin': origin,
            'destination': '',
            'price_in': instrument,
            'price': price,
            'price_w_expenses': price,
            'size': size,
            'commission': 0,
            'tax': 0,
            'total': total,
            'date_order': (date_order if date_order is not None else date_execution),
            'description': f'Dividend from {origin}.',
            'notes': notes,
            'commission_notes': '',
            'tax_notes': '',
            'account': account,
        }

        return self._add_rows(op_dividend)

    @validate_call(config={'arbitrary_types_allowed': True})
    def sell(
        self,
        account: str,
        date_execution: datetime.datetime,
        instrument: str,
        size: PositiveInt | PositiveFloat,
        price_in: str,
        price: NonNegativeInt | NonNegativeFloat,
        commission: NonNegativeInt | NonNegativeFloat = 0,
        tax: NonNegativeInt | NonNegativeFloat = 0,
        date_order: datetime.datetime | None = None,
        notes: str = '',
        commission_notes: str = '',
        tax_notes: str = '',
        stated_total: NonNegativeInt | NonNegativeFloat | None = None,
        tolerance_decimals: NonNegativeInt = 4,
    ):
        """Creates the operations needed to process a sell.

        Parameters
        ----------
        account : str
            Account used for the operations.
        date_execution : str
            Date when the operations are done.
        instrument : str
            The instrument that decreases its amount by the operations.
        size : PositiveInt | PositiveFloat
            The amount sold.
        price_in : str
            The instrument obtained by the sell.
        price : NonNegativeInt | NonNegativeFloat
            The current market price for the instrument.
        commission : NonNegativeInt | NonNegativeFloat, optional
            The amount payed to do the operation in `price_in`, by default 0.
        tax : NonNegativeInt | NonNegativeFloat, optional
            The amount payed in taxes in `price_in`, by default 0.
        date_order : str | None, optional
            Date when the operations are ordered. By default None, in which case `date_execution` is used.
        notes : str, optional
            Notes to add to the operations, by default ''.
        commission_notes : str, optional
            Notes about the commission to add to the operations, by default ''.
        tax_notes : str, optional
            Notes about the tax to add to the operations, by default ''.
        stated_total : NonNegativeInt  |  NonNegativeFloat  |  None, optional
            Used to verify if the inner calculation (`calculated_total = size * price + commission + tax`) is correct (`abs(stated_total - calculated_total) < 10**-tolerance_decimals`), By default None.

        Returns
        -------
        tuple[int, pd.DataFrame]
            Returns a tuple of the inserted opid and the pd.DataFrame inserted.

        Raises
        ------
        ValueError
            'When adding sell operation, the stated total is different from the calculated total. By more than the minimum of {10**-tolerance_decimals}.'
        """
        calculated_total = size * price - commission - tax

        # Allow inconsistencies up to 4 decimals
        if stated_total is not None and not (
            abs(stated_total - calculated_total) < 10**-tolerance_decimals
        ):
            error_msg = ' '.join(
                [
                    'When adding sell operation,',
                    'the stated total is different from the calculated total.',
                    f'By more than the minimum of {10**-tolerance_decimals}.',
                ]
            )
            error_dict = {
                'date_execution': date_execution,
                'instrument': instrument,
                'stated_total': stated_total,
                'size': size,
                'price': price,
                'commission': commission,
                'tax': tax,
                'calculated_total': calculated_total,
                'calculated_total-stated_total': calculated_total - stated_total,
            }
            raise ValueError(error_msg, error_dict)

        price_w_expenses = calculated_total / size

        op_1 = {}  # sell
        op_2 = {}  # uninvest
        op_1['date_execution'] = date_execution  # sell
        op_2['date_execution'] = date_execution  # uninvest
        op_1['operation'] = 'sell'  # sell
        op_2['operation'] = 'uninvest'  # uninvest
        op_1['instrument'] = instrument  # sell
        op_2['instrument'] = price_in  # uninvest
        op_1['origin'] = ''  # sell
        op_2['origin'] = instrument  # uninvest
        op_1['destination'] = price_in  # sell
        op_2['destination'] = ''  # uninvest
        op_1['price_in'] = price_in  # sell
        op_2['price_in'] = price_in  # uninvest
        op_1['price'] = price  # sell
        op_2['price'] = 1  # uninvest
        op_1['price_w_expenses'] = price_w_expenses  # sell
        op_2['price_w_expenses'] = 1
        op_1['size'] = -1 * size  # sell
        op_2['size'] = calculated_total  # uninvest
        op_1['commission'] = commission  # sell
        op_2['commission'] = 0  # uninvest
        op_1['tax'] = tax  # sell
        op_2['tax'] = 0  # uninvest
        op_1['total'] = -1 * calculated_total  # sell
        op_2['total'] = calculated_total  # uninvest
        op_1['date_order'] = date_order if date_order is not None else date_execution  # sell
        op_2['date_order'] = date_execution  # uninvest
        op_1['description'] = f'Sell {instrument}.'  # sell
        op_2['description'] = f'Uninvest in {instrument}.'  # uninvest
        op_1['notes'] = notes  # sell
        op_2['notes'] = notes  # uninvest
        op_1['commission_notes'] = commission_notes  # sell
        op_2['commission_notes'] = commission_notes  # uninvest
        op_1['tax_notes'] = tax_notes  # sell
        op_2['tax_notes'] = tax_notes  # uninvest
        op_1['account'] = account  # sell
        op_2['account'] = account  # uninvest

        return self._add_rows(data=[op_1, op_2])

    @validate_call(config={'arbitrary_types_allowed': True})
    def stock_dividend(
        self,
        account: str,
        date_execution: datetime.datetime,
        instrument: str,
        size: PositiveInt | PositiveFloat,
        date_order: datetime.datetime | None = None,
        notes: str = '',
    ):
        """Creates the operations needed to process a stock_dividend.

        Parameters
        ----------
        account : str
            Account used for the operations.
        date_execution : str
            Date when the operations are done.
        instrument : str
            The instrument that increases its amount by the operations.
        size : PositiveInt | PositiveFloat
            The amount obtained as dividend.
        date_order : str | None, optional
            Date when the operations are ordered. By default None, in which case `date_execution` is used.
        notes : str, optional
            Notes to add to the operations, by default ''.

        Returns
        -------
        tuple[int, pd.DataFrame]
            Returns a tuple of the inserted opid and the pd.DataFrame inserted.
        """
        op_stock_dividend = {
            'date_execution': date_execution,
            'operation': 'stock_dividend',
            'instrument': instrument,
            'origin': instrument,
            'destination': '',
            'price_in': instrument,
            'price': 0,
            'price_w_expenses': 0,
            'size': size,
            'commission': 0,
            'tax': 0,
            'total': 0,
            'date_order': (date_order if date_order is not None else date_execution),
            'description': f'Stock dividend from {instrument}.',
            'notes': notes,
            'commission_notes': '',
            'tax_notes': '',
            'account': account,
        }

        return self._add_rows(op_stock_dividend)

    @validate_call(config={'arbitrary_types_allowed': True})
    def withdraw(
        self,
        account: str,
        date_execution: datetime.datetime,
        instrument: str,
        size: PositiveInt | PositiveFloat,
        date_order: datetime.datetime | None = None,
        notes: str = '',
    ):
        """Creates the operations needed to process a withdraw.

        Important: Withdraw means a instrument exiting an account. A commission or tax for a withdraw is not possible, create a new operation for that purpose after doing the withdraw.

        Parameters
        ----------
        account : str
            Account used for the operations.
        date_execution : str
            Date when the operations are done.
        instrument : str
            The instrument that decreases its size by the operations.
        size : PositiveInt | PositiveFloat
            The amount withdrawn.
        date_order : str | None, optional
            Date when the operations are ordered. By default None, in which case `date_execution` is used.
        notes : str, optional
            Notes to add to the operations, by default ''.

        Returns
        -------
        tuple[int, pd.DataFrame]
            Returns a tuple of the inserted opid and the pd.DataFrame inserted.
        """
        op_withdraw = {
            'date_execution': date_execution,
            'operation': 'withdraw',
            'instrument': instrument,
            'origin': instrument,
            'destination': '',
            'price_in': instrument,
            'price': 1,
            'price_w_expenses': 1,
            'size': -size,
            'commission': 0,
            'tax': 0,
            'total': -size,
            'date_order': (date_order if date_order is not None else date_execution),
            'description': f'Withdraw {instrument}.',
            'notes': notes,
            'commission_notes': '',
            'tax_notes': '',
            'account': account,
        }

        return self._add_rows(op_withdraw)

    # *****************
    # MARK: METADATA
    # *****************

    @classmethod
    def get_ledger_columns(cls) -> Tuple[str]:
        """Returns the columns in a ledger.

        Returns
        -------
        List[str]
            Ledger columns.
        """
        return cls._ledger_columns

    @classmethod
    def get_operation_names(cls) -> List[str]:
        """Returns all the possible operation names in a ledger.

        Returns
        -------
        List[str]
            Sorted operation names in the ledger.
        """
        return sorted(cls._ops_names)

    def get_present_accounts(self) -> np.ndarray:
        """Returns present accounts in the ledger.

        Returns
        -------
        np.ndarray
            Present accounts in the ledger.
        """
        return self._ledger_df['account'].unique()

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

    def set_instruments_metadata(self, instruments_metadata: dict) -> None:
        """Set instruments' metadata.

        Parameters
        ----------
        instruments_metadata : dict
            The required format is the following:
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
        extraneous_instruments = set(instruments_metadata.keys()) - set(
            self.get_present_instruments()
        )
        if len(extraneous_instruments) > 0:
            warnings.warn(
                'Some instruments provided in the dict are not present in the ledger: '
                + f'{str(extraneous_instruments)}'
                + ' (still, storing all in the instruments metadata).',
                stacklevel=2,
            )
        for instr, metadata in instruments_metadata.items():
            if not isinstance(metadata, dict):
                continue
            namedict = {}
            typedict = {}
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

        dfs_to_concat = [pd.DataFrame(instruments_set, columns=['instrument'])]

        if instrument_type is True:
            rows_types = []
            for instr in instruments_set:
                metadata = self._instruments_metadata.get(instr)
                instr_type = metadata.get('type', '') if isinstance(metadata, dict) else ''
                rows_types.append([instr_type])
            dfs_to_concat.append(pd.DataFrame(rows_types, columns=['instrument_type']))

        if instrument_name is True:
            rows_names = []
            for instr in instruments_set:
                metadata = self._instruments_metadata.get(instr)
                instr_name = metadata.get('name', '') if isinstance(metadata, dict) else ''
                rows_names.append([instr_name])
            dfs_to_concat.append(pd.DataFrame(rows_names, columns=['instrument_name']))

        if instrument_in_ledger is True:
            extraneous_instruments = set(self._instruments_metadata.keys()) - set(
                self.get_present_instruments()
            )
            rows_extraneous = []
            for instr in instruments_set:
                is_extraneous = False if instr in extraneous_instruments else True
                rows_extraneous.append([is_extraneous])
            dfs_to_concat.append(pd.DataFrame(rows_extraneous, columns=['in_ledger']))

        to_return = (
            pd.concat(dfs_to_concat, join='inner', axis=1)
            .sort_values('instrument')
            .reset_index(drop=True)
        )

        return self.simplify_dtypes(to_return)

    # *****************
    # MARK: I/O
    # *****************

    def save(self, path: str, overwrite: bool = False):
        """Saves file to .spl extension, which includes the ledger and the instruments' metadata.

        The .spl extension is used to keep track of the files used by this class, however, these files are actually HDF files.

        Parameters
        ----------
        path : str
            File path.
        overwrite : bool, optional
            Whether to overwrite the file specified in path if it exists.
        """

        pathlib_path = pathlib.Path(path)

        if pathlib_path.is_dir():
            raise ValueError(f'path [{path}] is a directory: aborting.')

        if overwrite is False and pathlib_path.is_file():
            raise ValueError(f'path [{path}] exists, overwrite is False: aborting.')

        if pathlib_path.suffix != '.spl':
            warnings.warn(
                'Not using the .spl suffix. The .spl extension is recommended to keep track of this class\' files.',
                stacklevel=2,
            )

        ledger_df = self.ledger(
            instrument_type=False,
            instrument_name=False,
            operation_columns=False,
            instrument_cumsum_by_operation=False,
            operation_columns_balance=False,
            thousands_fmt_sep=False,
            thousands_fmt_decimals=1,
        )

        # From the documentation it appears that compression can be done in `pd.HDFStore()`
        #  but from testing, even if the parameters exist, file size wasn't getting smaller
        #  it even grew. So compression is done in `store.put()`.
        with pd.HDFStore(pathlib_path, mode='w') as store:
            store.put(
                '_ledger_df',
                ledger_df,
                format='table',
                complevel=9,
                complib='zlib',
            )
            store.get_storer('_ledger_df').attrs._instruments_metadata = self._instruments_metadata

    def load(
        self,
        path,
    ):
        """Loads a file created by this class, ideally with an .spl extension.

        Parameters
        ----------
        path : _type_
            File path.
        """
        with pd.HDFStore(path, mode='r') as store:
            self._ledger_df = store.get('_ledger_df')
            self._instruments_metadata = store.get_storer('_ledger_df').attrs._instruments_metadata

    def to_excel(
        self,
        path: str,
        instrument_type: bool = False,
        instrument_name: bool = False,
        operation_columns: bool = False,
        instrument_cumsum_by_operation: bool = False,
        operation_columns_balance: bool = False,
        overwrite: bool = False,
    ):
        """Saves the ledger to an Excel file.

        Parameters
        ----------
        path : str
            File path.
        instrument_type : bool, optional
            Wether to add a column for the instrument type. By default False.
        instrument_name : bool, optional
            Wether to add a column for the instrument name. By default False.
        operation_columns : bool, optional
            Whether to add operation_columns to The Ledger. By default False.
        instrument_cumsum_by_operation : bool, optional
            Whether to add instrument_cumsum_by_operation to The Ledger. By default False.
        operation_columns_balance : bool, optional
            Whether to add operation_columns_balance to The Ledger. By default False.
        overwrite : bool, optional
            Whether to overwrite the file specified in path if it exists. by default False.

        Raises
        ------
        ValueError
            path [{path}] is a directory: aborting.
        ValueError
            path [{path}] exists, overwrite is False: aborting.
        """
        pathlib_path = pathlib.Path(path)

        if pathlib_path.is_dir():
            raise ValueError(f'path [{path}] is a directory: aborting.')

        if overwrite is False and pathlib_path.is_file():
            raise ValueError(f'path [{path}] exists, overwrite is False: aborting.')

        # From https://xlsxwriter.readthedocs.io/example_pandas_autofilter.html

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(path, engine="xlsxwriter")

        show_index = True
        add_if_show_index = 1 if show_index is True else 0

        # Convert the DataFrame to an XlsxWriter Excel object
        self.ledger(
            instrument_type=instrument_type,
            instrument_name=instrument_name,
            operation_columns=operation_columns,
            instrument_cumsum_by_operation=instrument_cumsum_by_operation,
            operation_columns_balance=operation_columns_balance,
        ).to_excel(
            writer,
            sheet_name="ledger",
            index=show_index,
        )

        # Get the xlsxwriter workbook and worksheet objects.
        # workbook = writer.book
        worksheet = writer.sheets["ledger"]

        # Get the dimensions of the DataFrame.
        (max_row, max_col) = self.ledger(
            instrument_type=instrument_type,
            instrument_name=instrument_name,
            operation_columns=operation_columns,
            instrument_cumsum_by_operation=instrument_cumsum_by_operation,
            operation_columns_balance=operation_columns_balance,
        ).shape

        # Set the autofilter.
        # TODO: Untested, to add a filter to the first column
        worksheet.autofilter(0, 0, max_row, max_col)

        # From https://xlsxwriter.readthedocs.io/example_panes.html
        worksheet.freeze_panes(1, 8 + add_if_show_index)

        # From https://stackoverflow.com/a/75120836/1071459
        worksheet.autofit()

        # Close the Pandas Excel writer and output the Excel file.
        writer.close()
