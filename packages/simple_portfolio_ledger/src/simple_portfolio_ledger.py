'''A simple ledger to keep track of a Portfolio's movements.'''

import functools
import inspect
import warnings
import pathlib
from typing import List

import numpy as np
import pandas as pd

# Note: MARK is a functionality from Visual Studio Code to find things in the minimap.


class SimplePortfolioLedger:

    _ledger_columns = (
        'opid',
        'subopid',
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

    def ledger(
        self,
        instrument_type: bool = False,
        instrument_name: bool = False,
        cols_operation: bool = False,
        cols_operation_cumsum: bool = False,
        cols_operation_balance_by_instrument: bool = False,
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

        Returns
        -------
        pd.DataFrame
            The Ledger, with optional additional information.
        """
        if len(self._ledger_df) == 0:
            warnings.warn('WARNING: Ledger is empty, showing only basic structure.', stacklevel=2)

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

        dfs_toconcat = [the_ledger]

        if cols_operation is True:
            tmp = self.cols_operation(show_instr_accnt=False)
            dfs_toconcat.append(tmp)
        if cols_operation_cumsum is True:
            tmp = self.cols_operation_cumsum(show_instr_accnt=False)
            dfs_toconcat.append(tmp)
        if cols_operation_balance_by_instrument is True:
            tmp = self.cols_operation_balance_by_instrument(show_instr_accnt=False)
            dfs_toconcat.append(tmp)

        to_return = (
            pd
            # Join DataFrames
            .concat(dfs_toconcat, join='inner', axis=1)
            # Convert dates to appropriate format
            .astype(
                {
                    'date_execution': 'datetime64[ns]',
                    'date_order': 'datetime64[ns]',
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
    def cols_operation(self, show_instr_accnt=False) -> pd.DataFrame:
        """Returns a dataframe with 1 column per operation.

        Parameters
        ----------
        show_instr_accnt : bool, optional
            Whether or not to show the instrument and the account. By default False.

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
                    # Remove name for columns' index
                    .rename_axis(None, axis=1)
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

                to_return = self.simplify_dtypes(to_return)

        if show_instr_accnt is True:
            return to_return[['instrument', 'account', *sorted(self._ops_names)]]
        else:
            return to_return[[*sorted(self._ops_names)]]

    @_deco_check_ledger_for_cols
    def cols_operation_cumsum(self, show_instr_accnt=False) -> pd.DataFrame:
        """Returns a DataFrame with one column per operation but do a cumsum per instrument/account.

        Parameters
        ----------
        show_instr_accnt : bool, optional
            Whether or not to show the instrument and the account. By default False.

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
                    'instrument',
                    'account',
                    *ops_cumsum_names,
                    'cumsum held',
                    'cumsum invested',
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
                    # Remove name for columns' index
                    .rename_axis(None, axis=1)
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
                        invested=(
                            lambda x: x[['cumsum invest', 'cumsum uninvest']].sum(axis=1).abs()
                        ),
                    )
                    # Rename newly created columns
                    .rename(columns={'held': 'cumsum held', 'invested': 'cumsum invested'})
                    # Sort by the original index
                    .sort_index()
                    # Try to pass colums where dtype is object to a type like int64 or float64
                    .infer_objects()
                )

                to_return = self.simplify_dtypes(to_return)

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
    def cols_operation_balance_by_instrument(self, show_instr_accnt=False) -> pd.DataFrame:
        """Returns a DataFrame with a balance per operation per instrument/account.

        Parameters
        ----------
        show_instr_accnt : bool, optional
            Whether or not to show the instrument and the account. By default False.

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

            to_return = self.simplify_dtypes(to_return)

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

    def buy(
        self,
        date_execution,
        instrument,
        price_in,
        price,
        size,
        account,
        commission=0,
        tax=0,
        stated_total=None,  # Used to verify if inner calculation is correct
        date_order=None,
        notes='',
        commission_notes='',
        tax_notes='',
        tolerance_decimals=4,
    ):
        """
        REMOVE BUT FORMAT LATER: All values are positive. Sign is changed inside the function.
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
        op_1['stated_total'] = -1 * calculated_total  # invest
        op_2['stated_total'] = calculated_total  # buy
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

    def deposit(
        self,
        date_execution,
        instrument,
        amount,
        account,
        date_order=None,
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
        op_deposit = {
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
            'description': f'Deposit {instrument}.',
            'notes': notes,
            'commission_notes': '',
            'tax_notes': '',
            'account': account,
        }

        return self._add_rows(op_deposit)

    def dividend(
        self,
        date_execution,
        instrument_from,
        instrument_received,
        amount,
        account,
        date_order=None,
        notes='',
    ):
        size = self._ledger_df.query(
            f'instrument == "{instrument_from}" and date_execution <= "{date_execution}"'
        )['size'].sum()

        if size == 0:
            raise ValueError(
                ' '.join(
                    [
                        'When doing operation "dividend",',
                        f'the calculated size of held instrument {instrument_from} is 0.',
                        'This could mean there are no previous operations for the instrument',
                        'or the sum of all operations is 0. Check the ledger.',
                    ]
                )
            )

        price = amount / size
        op_dividend = {
            'date_execution': date_execution,
            'operation': 'dividend',
            'instrument': instrument_received,
            'origin': instrument_from,
            'destination': '',
            'price_in': instrument_received,
            'price': price,
            'price_w_expenses': price,
            'size': size,
            'commission': 0,
            'tax': 0,
            'stated_total': amount,
            'date_order': (date_order if date_order is not None else date_execution),
            'description': f'Dividend from {instrument_from}.',
            'notes': notes,
            'commission_notes': '',
            'tax_notes': '',
            'account': account,
        }

        return self._add_rows(op_dividend)

    def sell(
        self,
        date_execution,
        instrument,
        price_in,
        price,
        size,
        account,
        commission=0,
        tax=0,
        stated_total=None,  # Used to verify if inner calculation is correct
        date_order=None,
        notes='',
        commission_notes='',
        tax_notes='',
        tolerance_decimals=4,
    ):
        """
        REMOVE BUT FORMAT LATER: All values are positive. Sign is changed inside the function.
        """
        calculated_total = size * price - commission - tax

        # Allow inconsistencies up to 4 decimals
        if stated_total is not None and not (
            abs(stated_total - calculated_total) < 10**-tolerance_decimals
        ):
            error_msg = ''.join(
                [
                    'When adding sell operation, '
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
        op_1['stated_total'] = -1 * calculated_total  # sell
        op_2['stated_total'] = calculated_total  # uninvest
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

    def stock_dividend(
        self,
        date_execution,
        instrument,
        price_in,
        size,
        account,
        date_order=None,
        notes='',
    ):
        op_stock_dividend = {
            'date_execution': date_execution,
            'operation': 'stock dividend',
            'instrument': instrument,
            'origin': instrument,
            'destination': '',
            'price_in': price_in,
            'price': 0,
            'price_w_expenses': 0,
            'size': size,
            'commission': 0,
            'tax': 0,
            'stated_total': 0,
            'date_order': (date_order if date_order is not None else date_execution),
            'description': f'Stock dividend from {instrument}.',
            'notes': notes,
            'commission_notes': '',
            'tax_notes': '',
            'account': account,
        }

        return self._add_rows(op_stock_dividend)

    def withdraw(
        self,
        date_execution,
        instrument,
        amount,
        account,
        date_order=None,
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
        op_withdraw = {
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

        to_return = (
            pd.concat(dfs_toconcat, join='inner', axis=1)
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
            cols_operation=False,
            cols_operation_cumsum=False,
            cols_operation_balance_by_instrument=False,
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
        cols_operation: bool = False,
        cols_operation_cumsum: bool = False,
        cols_operation_balance_by_instrument: bool = False,
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
        cols_operation : bool, optional
            Whether to add cols_operation to The Ledger. By default False.
        cols_operation_cumsum : bool, optional
            Whether to add cols_operation_cumsum to The Ledger. By default False.
        cols_operation_balance_by_instrument : bool, optional
            Whether to add cols_operation_balance_by_instrument to The Ledger. By default False.
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
            cols_operation=cols_operation,
            cols_operation_cumsum=cols_operation_cumsum,
            cols_operation_balance_by_instrument=cols_operation_balance_by_instrument,
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
            cols_operation=cols_operation,
            cols_operation_cumsum=cols_operation_cumsum,
            cols_operation_balance_by_instrument=cols_operation_balance_by_instrument,
        ).shape

        # Set the autofilter.
        worksheet.autofilter(0, 1, max_row, max_col)

        # From https://xlsxwriter.readthedocs.io/example_panes.html
        worksheet.freeze_panes(1, 3 + add_if_show_index)

        # From https://stackoverflow.com/a/75120836/1071459
        worksheet.autofit()

        # Close the Pandas Excel writer and output the Excel file.
        writer.close()
