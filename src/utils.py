import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import contextily as cx

from matplotlib.patches import Patch
from ipywidgets import interact, widgets

def load_fire_data():
    """
    Load and return all processed dataframes for Coon, Hayman, and Carr fires.
    Returns:
        - coon_df: Masked DataFrame for Coon Fire (2000-04-30 to 2001-06-30)
        - hayman_df: Masked DataFrame for Hayman Fire (year 2003)
        - carr_dfs: List of masked DataFrames [brandy, boulder, whiskey] for 2018-10-01 to 2019-09-30
        - carr_ws: List of Carr sub-watershed names
        - area_dict: Dictionary of watershed areas (km^2)
    """
    # Load COON
    coon_df = pd.read_csv('src/coon/totalwatsed2.csv', skiprows=4)
    coon_df['Date'] = pd.to_datetime(coon_df[['Year', 'Month', 'Day']])
    coon_df = coon_df[(coon_df['Date'] >= '2000-04-30') & (coon_df['Date'] <= '2001-06-30')]

    # Load HAYMAN
    hayman_df = pd.read_csv('src/hayman/totalwatsed2.csv', skiprows=4)
    hayman_df['Date'] = pd.to_datetime(hayman_df[['Year', 'Month', 'Day']])
    hayman_df = hayman_df[hayman_df['Year'] == 2003]

    # Load CARR
    carr_ws = ['brandy', 'boulder', 'whiskey']
    carr_dfs = []
    for ws in carr_ws:
        path = f'src/carr/{ws}/totalwatsed2.csv'
        df = pd.read_csv(path, skiprows=4)
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df = df[(df['Date'] >= '2018-10-01') & (df['Date'] <= '2019-09-30')]
        carr_dfs.append(df)

    area_dict = {
        'hayman': 3.22,
        'coon': 3.99,
        'carr': [23.2, 12.6, 22.1]  # brandy, boulder, whiskey
    }

    return coon_df, hayman_df, carr_dfs, carr_ws, area_dict

def compute_sediment_yield(coon_df, hayman_df, carr_dfs, carr_ws, area_dict):
    """
    Compute sediment yield (tonnes/km^2) from Coon, Hayman, and Carr fire data.
    Returns:
        - modeled_sed_yield: List of yields [Coon, Carr: Brandy, Carr: Boulder, Carr: Whiskey, Hayman]
        - fire_names: Corresponding fire/watershed labels
    """
    # COON
    coon_sed_yield = coon_df['Sed Del Density (tonne/ha)'].sum() * 100

    # HAYMAN
    hayman_sed_yield = (
        hayman_df['Cumulative Sed Del (tonnes)'].max() -
        hayman_df['Cumulative Sed Del (tonnes)'].min()
    ) / area_dict['hayman']

    # CARR
    carr_sed_yield = []
    for i, df in enumerate(carr_dfs):
        yield_density = df['Sed Del Density (tonne/ha)'].sum() * 100
        carr_sed_yield.append(yield_density)

    # Combine
    modeled_sed_yield = [coon_sed_yield] + carr_sed_yield + [hayman_sed_yield]
    fire_names = ['Coon'] + [f'Carr: {ws.capitalize()}' for ws in carr_ws] + ['Hayman']

    return modeled_sed_yield, fire_names

def plot_observed_sediment_yield():
    """
    Plot observed sediment yield with error bars by watershed and fire.
    """
    data = {
        'fire': ['coon', 'carr', 'carr', 'carr', 'hayman'],
        'watershed': ['workmen', 'brandy', 'boulder', 'whiskey', 'saloon'],
        'sediment_totals': [22, 4080, 2700, 305, 1300],
        'lower': [22, 3480, 2200, 245, 300],
        'upper': [36, 4680, 3200, 365, 2300],
        'units': ['t_km^2']*5,
        'time_period': [1.6, 1.0, 1.0, 1.0, 1.0]
    }

    df = pd.DataFrame(data)
    df['error_lower'] = df['sediment_totals'] - df['lower']
    df['error_upper'] = df['upper'] - df['sediment_totals']
    error = [df['error_lower'], df['error_upper']]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['watershed'], df['sediment_totals'], yerr=error, capsize=5)

    colors = {'coon': 'skyblue', 'carr': 'salmon', 'hayman': 'lightgreen'}
    for bar, fire in zip(bars, df['fire']):
        bar.set_color(colors[fire])

    ax.set_ylabel('Sediment Yield (tonnes/km²)')
    ax.set_title('Observed Sediment Yield by Watershed')
    ax.set_xlabel('Watershed')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    legend_handles = [Patch(color=clr, label=fire) for fire, clr in colors.items()]
    ax.legend(handles=legend_handles, title='Fire')

    plt.tight_layout()
    plt.show()

def plot_modeled_vs_observed(modeled_sed_yield):
    """
    Plot a scatter plot comparing modeled and observed sediment yield with error bars.
    """
    obs = pd.DataFrame({
        'sediment_totals': [22, 4080, 2700, 305, 1300],
        'lower': [22, 3480, 2200, 245, 300],
        'upper': [36, 4680, 3200, 365, 2300]
    })
    labels = ['Coon', 'Carr: Brandy', 'Carr: Boulder', 'Carr: Whiskey', 'Hayman']

    fig, ax = plt.subplots(figsize=(8, 8))

    xerr_lower = obs['sediment_totals'] - obs['lower']
    xerr_upper = obs['upper'] - obs['sediment_totals']
    xerr = [xerr_lower, xerr_upper]

    ax.errorbar(
        obs['sediment_totals'],
        modeled_sed_yield,
        xerr=xerr,
        fmt='o',
        color='steelblue',
        ecolor='gray',
        capsize=5,
        markersize=10,
        markeredgecolor='k',
        zorder=3,
        label='Observed ± uncertainty'
    )

    for x, y, label in zip(obs['sediment_totals'], modeled_sed_yield, labels):
        ax.text(x, y, label, fontsize=10, ha='left', va='bottom')

    min_val = min(min(obs['sediment_totals']), min(modeled_sed_yield))
    max_val = max(max(obs['sediment_totals']), max(modeled_sed_yield))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line', zorder=2)

    ax.set_xlabel('Observed Sediment Yield (tonnes/km²)', fontsize=12)
    ax.set_ylabel('Modeled Sediment Yield (tonnes/km²)', fontsize=12)
    ax.set_title('Modeled vs. Observed Sediment Yield', fontsize=14)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.tight_layout()
    plt.show()

def interactive_hayman_plot(hayman_df):
    """
    Interactive plot for Hayman Fire 2003 data with selectable runoff variable.
    """
    df_2003 = hayman_df.copy()
    df_2003['Date'] = pd.to_datetime(df_2003[['Year', 'Month', 'Day']])
    df_2003 = df_2003[df_2003['Date'].dt.year == 2003].copy()
    df_2003['Cumulative Sed Del Density (t/ha)'] = df_2003['Sed Del Density (tonne/ha)'].cumsum()

    def plot_with_variable(runoff_var):
        fig, ax1 = plt.subplots(figsize=(14, 6))

        ax1.plot(df_2003['Date'], df_2003[runoff_var], label=runoff_var, color='blue')
        ax1.set_ylabel(runoff_var, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(df_2003['Date'], df_2003['Cumulative Sed Del Density (t/ha)'], color='red')
        ax2.set_ylabel('Cumulative Sediment Density (tonnes/ha)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('axes', 1.15))
        ax3.fill_between(df_2003['Date'], df_2003['Precipitation (mm)'], color='gray', alpha=0.3)
        ax3.set_ylabel('Rainfall (mm)', color='gray')
        ax3.tick_params(axis='y', labelcolor='gray')
        ax3.invert_yaxis()

        ax1.set_xlabel('Date')
        plt.title(f'{runoff_var} vs. Cumulative Sediment and Rainfall (Hayman 2003)')
        fig.tight_layout()
        plt.show()

    allowed_vars = [
        'Sed Del (kg)', 'Sed Del c1 (kg)', 'Sed Del c2 (kg)', 'Sed Del c3 (kg)',
        'Sed Del c4 (kg)', 'Sed Del c5 (kg)', 'Cumulative Sed Del (tonnes)',
        'Sed Del Density (tonne/ha)', 'Precipitation (mm)', 'Rain + Melt (mm)',
        'Transpiration (mm)', 'Evaporation (mm)', 'ET (mm)', 'Percolation (mm)',
        'Runoff (mm)', 'Lateral Flow (mm)', 'Storage (mm)',
        'Reservoir Volume (mm)', 'Baseflow (mm)', 'Aquifer Losses (mm)',
        'Streamflow (mm)'
        ]
    runoff_options = [col for col in allowed_vars if col in df_2003.columns]
    interact(plot_with_variable, runoff_var=widgets.Dropdown(options=runoff_options, description='Variable:'))

def interactive_hayman_map():
    """
    Interactive map to select and plot Hayman subcatchment polygons by attribute column.
    """
    hayman_gdf = gpd.read_file('src/hayman/arcmap/Saloon.gpkg', layer='subcatchments')
    hayman_web = hayman_gdf.to_crs(epsg=3857)

    def plot_column(column):
        fig, ax = plt.subplots(figsize=(12, 12))
        is_numeric = pd.api.types.is_numeric_dtype(hayman_web[column])

        # Set legend_kwds only for numeric columns
        legend_kwds = {
            'orientation': 'horizontal',
            'shrink': 0.7,
            'pad': 0.05,
            'aspect': 30
        } if is_numeric else {}

        # Plot with conditional legend settings
        hayman_web.plot(
            column=column,
            ax=ax,
            alpha=0.7,
            edgecolor='k',
            legend=True,
            legend_kwds=legend_kwds
        )

        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery)
        ax.set_axis_off()
        plt.title(f"Hayman Subcatchments Colored by '{column}'")
        plt.tight_layout()
        plt.show()

    allowed_columns = [
    'slope_scalar', 'length_m', 'width_m', 'direction',
    'aspect', 'area_m2', 'cancov', 'inrcov', 'rilcov', 'disturbed_class',
    'mukey', 'clay', 'sand', 'Runoff_Volume_m3', 'Subrunoff_Volume_m3',
    'Baseflow_Volume_m3', 'Soil_Loss_kg', 'Sediment_Deposition_kg',
    'Sediment_Yield_kg', 'Solub_React_Phosphorus_kg',
    'Particulate_Phosphorus_kg', 'Total_Phosphorus_kg', 'Soil', 'Runoff_mm',
    'Subrunoff_mm', 'Baseflow_mm', 'DepLoss_kg'
    ]
    column_options = [col for col in allowed_columns if col in hayman_web.columns]
    interact(plot_column, column=widgets.Dropdown(options=column_options, description='Attribute:'))

def interactive_coon_plot(coon_df):
    """
    Interactive plot for Coon Fire data with selectable runoff variable.
    """
    df = coon_df.copy()
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df = df[(df['Date'] >= '2000-04-30') & (df['Date'] <= '2001-06-30')].copy()
    df['Cumulative Sed Del Density (t/ha)'] = df['Sed Del Density (tonne/ha)'].cumsum()

    def plot_with_variable(runoff_var):
        fig, ax1 = plt.subplots(figsize=(14, 6))

        ax1.plot(df['Date'], df[runoff_var], label='Runoff', color='blue')
        ax1.set_ylabel(runoff_var, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(df['Date'], df['Cumulative Sed Del Density (t/ha)'], color='red')
        ax2.set_ylabel('Cumulative Sediment Density (tonnes/ha)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('axes', 1.15))
        ax3.fill_between(df['Date'], df['Precipitation (mm)'], color='gray', alpha=0.3)
        ax3.set_ylabel('Rainfall (mm)', color='gray')
        ax3.tick_params(axis='y', labelcolor='gray')
        ax3.invert_yaxis()

        ax1.set_xlabel('Date')
        plt.title(f'{runoff_var} vs. Cumulative Sediment and Rainfall (Coon 2000–2001)')
        fig.tight_layout()
        plt.show()

    allowed_vars = [
    'Sed Del (kg)', 'Sed Del c1 (kg)', 'Sed Del c2 (kg)', 'Sed Del c3 (kg)',
    'Sed Del c4 (kg)', 'Sed Del c5 (kg)', 'Cumulative Sed Del (tonnes)',
    'Sed Del Density (tonne/ha)', 'Precipitation (mm)', 'Rain + Melt (mm)',
    'Transpiration (mm)', 'Evaporation (mm)', 'ET (mm)', 'Percolation (mm)',
    'Runoff (mm)', 'Lateral Flow (mm)', 'Storage (mm)',
    'Reservoir Volume (mm)', 'Baseflow (mm)', 'Aquifer Losses (mm)',
    'Streamflow (mm)'
    ]
    runoff_options = [col for col in allowed_vars if col in df.columns]
    interact(plot_with_variable, runoff_var=widgets.Dropdown(options=runoff_options, description='Variable:'))

def interactive_coon_map():
    """
    Interactive map to select and plot Coon Fire subcatchment polygons by attribute column.
    """
    coon_gdf = gpd.read_file('src/coon/arcmap/Workman_main.gpkg', layer='subcatchments')
    coon_web = coon_gdf.to_crs(epsg=3857)

    def plot_column(column):
        fig, ax = plt.subplots(figsize=(12, 12))
        is_numeric = pd.api.types.is_numeric_dtype(coon_web[column])
        legend_kwds = {
            'orientation': 'horizontal',
            'shrink': 0.7,
            'pad': 0.05,
            'aspect': 30
        } if is_numeric else {}

        coon_web.plot(
            column=column,
            ax=ax,
            alpha=0.7,
            edgecolor='k',
            legend=True,
            legend_kwds=legend_kwds
        )

        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery)
        ax.set_axis_off()
        plt.title(f"Coon Subcatchments Colored by '{column}'")
        plt.tight_layout()
        plt.show()

    allowed_columns = [
        'slope_scalar', 'length_m', 'width_m', 'direction',
        'aspect', 'area_m2', 'cancov', 'inrcov', 'rilcov', 'disturbed_class',
        'mukey', 'clay', 'sand', 'Runoff_Volume_m3', 'Subrunoff_Volume_m3',
        'Baseflow_Volume_m3', 'Soil_Loss_kg', 'Sediment_Deposition_kg',
        'Sediment_Yield_kg', 'Solub_React_Phosphorus_kg',
        'Particulate_Phosphorus_kg', 'Total_Phosphorus_kg', 'Soil', 'Runoff_mm',
        'Subrunoff_mm', 'Baseflow_mm', 'DepLoss_kg'
        ]
    column_options = [col for col in allowed_columns if col in coon_web.columns]
    interact(plot_column, column=widgets.Dropdown(options=column_options, description='Attribute:'))

def interactive_carr_plot(carr_dfs, carr_ws):
    """
    Interactive plot for Carr Fire watersheds with selectable watershed and variable.
    """
    def plot_watershed(watershed):
        idx = carr_ws.index(watershed)
        df = carr_dfs[idx].copy()
        df['Cumulative Sed Del Density (t/ha)'] = df['Sed Del Density (tonne/ha)'].cumsum()

        def plot_with_variable(runoff_var):
            fig, ax1 = plt.subplots(figsize=(14, 6))
            ax1.plot(df['Date'], df[runoff_var], label=runoff_var, color='blue')
            ax1.set_ylabel(runoff_var, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            ax2 = ax1.twinx()
            ax2.plot(df['Date'], df['Cumulative Sed Del Density (t/ha)'], color='red')
            ax2.set_ylabel('Cumulative Sediment Density (tonnes/ha)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('axes', 1.15))
            ax3.fill_between(df['Date'], df['Precipitation (mm)'], color='gray', alpha=0.3)
            ax3.set_ylabel('Rainfall (mm)', color='gray')
            ax3.tick_params(axis='y', labelcolor='gray')
            ax3.invert_yaxis()

            ax1.set_xlabel('Date')
            plt.title(f'{runoff_var} vs. Cumulative Sediment and Rainfall (Carr: {watershed.title()})')
            fig.tight_layout()
            plt.show()

        allowed_vars = [
            'Sed Del (kg)', 'Sed Del c1 (kg)', 'Sed Del c2 (kg)', 'Sed Del c3 (kg)',
            'Sed Del c4 (kg)', 'Sed Del c5 (kg)', 'Cumulative Sed Del (tonnes)',
            'Sed Del Density (tonne/ha)', 'Precipitation (mm)', 'Rain + Melt (mm)',
            'Transpiration (mm)', 'Evaporation (mm)', 'ET (mm)', 'Percolation (mm)',
            'Runoff (mm)', 'Lateral Flow (mm)', 'Storage (mm)',
            'Reservoir Volume (mm)', 'Baseflow (mm)', 'Aquifer Losses (mm)',
            'Streamflow (mm)'
        ]
        runoff_options = [col for col in allowed_vars if col in df.columns]
        interact(plot_with_variable, runoff_var=widgets.Dropdown(options=runoff_options, description='Variable:'))

    interact(plot_watershed, watershed=widgets.Dropdown(options=carr_ws, description='Watershed:'))

def interactive_carr_map():
    """
    Interactive map to select Carr Fire watershed and attribute column to display.
    """
    carr_ws = ['brandy', 'boulder', 'whiskey']
    gdf_dict = {
        ws: gpd.read_file(f'src/carr/{ws}/arcmap/{ws}.gpkg', layer='subcatchments').to_crs(epsg=3857)
        for ws in carr_ws
    }

    def plot_carr(watershed, column):
        gdf = gdf_dict[watershed]
        fig, ax = plt.subplots(figsize=(16, 16))
        is_numeric = pd.api.types.is_numeric_dtype(gdf[column])
        legend_kwds = {
            'orientation': 'horizontal',
            'shrink': 0.7,
            'pad': 0.05,
            'aspect': 30
        } if is_numeric else {}

        gdf.plot(
            column=column,
            ax=ax,
            alpha=0.7,
            edgecolor='k',
            legend=True,
            legend_kwds=legend_kwds
        )
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery)
        ax.set_axis_off()
        plt.title(f"Carr ({watershed.title()}) Subcatchments Colored by '{column}'")
        plt.tight_layout()
        plt.show()

    def update_columns(watershed):
        gdf = gdf_dict[watershed]
        allowed_columns = [
        'slope_scalar', 'length_m', 'width_m', 'direction',
        'aspect', 'area_m2', 'cancov', 'inrcov', 'rilcov', 'disturbed_class',
        'mukey', 'clay', 'sand', 'Runoff_Volume_m3', 'Subrunoff_Volume_m3',
        'Baseflow_Volume_m3', 'Soil_Loss_kg', 'Sediment_Deposition_kg',
        'Sediment_Yield_kg', 'Solub_React_Phosphorus_kg',
        'Particulate_Phosphorus_kg', 'Total_Phosphorus_kg', 'Soil', 'Runoff_mm',
        'Subrunoff_mm', 'Baseflow_mm', 'DepLoss_kg'
        ]
        column_options = [col for col in allowed_columns if col in gdf.columns]
        interact(lambda column: plot_carr(watershed, column), column=widgets.Dropdown(options=column_options, description='Attribute:'))

    interact(update_columns, watershed=widgets.Dropdown(options=carr_ws, description='Watershed:'))


