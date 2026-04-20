import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import time
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class OpenF1DataCollector:
    """
    Collects data from OpenF1 API
    """
    def __init__(self):
        self.base_url = "https://api.openf1.org/v1"
        
    def get_sessions(self, year=2024, session_type=None):
        """Get session information for a given year"""
        url = f"{self.base_url}/sessions"
        params = {"year": year}
        if session_type:
            params["session_type"] = session_type
            
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except requests.RequestException as e:
            print(f"Error fetching sessions: {e}")
            return pd.DataFrame()
    
    def get_drivers(self, year=2024):
        """Get driver information for a given year"""
        url = f"{self.base_url}/drivers"
        params = {"year": year}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except requests.RequestException as e:
            print(f"Error fetching drivers: {e}")
            return pd.DataFrame()
    
    def get_results(self, session_key):
        """Get race results for a specific session"""
        url = f"{self.base_url}/results"
        params = {"session_key": session_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except requests.RequestException as e:
            print(f"Error fetching results for session {session_key}: {e}")
            return pd.DataFrame()
    
    def get_pit_stops(self, session_key):
        """Get pit stop data for a specific session"""
        url = f"{self.base_url}/pit"
        params = {"session_key": session_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except requests.RequestException as e:
            print(f"Error fetching pit stops for session {session_key}: {e}")
            return pd.DataFrame()
    
    def get_positions(self, session_key):
        """Get position data for a specific session"""
        url = f"{self.base_url}/position"
        params = {"session_key": session_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except requests.RequestException as e:
            print(f"Error fetching positions for session {session_key}: {e}")
            return pd.DataFrame()
    
    def get_weather(self, session_key):
        """Get weather data for a specific session"""
        url = f"{self.base_url}/weather"
        params = {"session_key": session_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except requests.RequestException as e:
            print(f"Error fetching weather for session {session_key}: {e}")
            return pd.DataFrame()

class PitStopPredictor:
    """
    Predicts pit stop strategy and timing
    """
    def __init__(self):
        self.tire_compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
        self.tire_degradation = {'SOFT': 0.15, 'MEDIUM': 0.08, 'HARD': 0.05, 'INTERMEDIATE': 0.12, 'WET': 0.10}
        self.tire_performance = {'SOFT': 1.0, 'MEDIUM': 0.85, 'HARD': 0.75, 'INTERMEDIATE': 0.6, 'WET': 0.5}
        
    def predict_pit_strategy(self, race_distance, weather_conditions, tire_rules=True):
        """
        Predict optimal pit stop strategy
        """
        strategies = []
        
        # Determine tire compound based on weather
        if weather_conditions.get('rainfall', 0) > 0.5:
            primary_compound = 'INTERMEDIATE' if weather_conditions['rainfall'] < 2 else 'WET'
        else:
            temp = weather_conditions.get('track_temperature', 35)
            if temp > 40:
                primary_compound = 'HARD'
            elif temp > 30:
                primary_compound = 'MEDIUM'
            else:
                primary_compound = 'SOFT'
        
        # One-stop strategy
        pit_lap_1_stop = max(25, race_distance * 0.6)
        strategies.append({
            'strategy': '1-stop',
            'pit_laps': [pit_lap_1_stop],
            'tire_compounds': [primary_compound, self._get_second_compound(primary_compound)],
            'estimated_time_loss': 25.0,  # seconds
            'risk_level': 'Medium'
        })
        
        # Two-stop strategy
        pit_lap_2_stop_1 = max(15, race_distance * 0.35)
        pit_lap_2_stop_2 = max(35, race_distance * 0.75)
        strategies.append({
            'strategy': '2-stop',
            'pit_laps': [pit_lap_2_stop_1, pit_lap_2_stop_2],
            'tire_compounds': ['SOFT', 'MEDIUM', 'HARD'],
            'estimated_time_loss': 50.0,  # seconds
            'risk_level': 'Low'
        })
        
        # Three-stop strategy (aggressive)
        if race_distance > 50:
            strategies.append({
                'strategy': '3-stop',
                'pit_laps': [race_distance * 0.25, race_distance * 0.5, race_distance * 0.75],
                'tire_compounds': ['SOFT', 'SOFT', 'MEDIUM', 'HARD'],
                'estimated_time_loss': 75.0,  # seconds
                'risk_level': 'High'
            })
        
        return strategies
    
    def _get_second_compound(self, first_compound):
        """Get complementary tire compound"""
        if first_compound == 'SOFT':
            return 'MEDIUM'
        elif first_compound == 'MEDIUM':
            return 'HARD'
        else:
            return 'MEDIUM'
    
    def analyze_pit_stop_data(self, pit_stops_df):
        """
        Analyze historical pit stop data
        """
        if pit_stops_df.empty:
            return {}
        
        analysis = {
            'avg_pit_time': pit_stops_df['pit_duration'].mean() if 'pit_duration' in pit_stops_df.columns else 25.0,
            'fastest_pit': pit_stops_df['pit_duration'].min() if 'pit_duration' in pit_stops_df.columns else 20.0,
            'slowest_pit': pit_stops_df['pit_duration'].max() if 'pit_duration' in pit_stops_df.columns else 35.0,
            'total_pit_stops': len(pit_stops_df),
            'avg_pit_lap': pit_stops_df['lap_number'].mean() if 'lap_number' in pit_stops_df.columns else 30
        }
        
        return analysis

class F1VisualizationEngine:
    """
    Creates comprehensive F1 visualizations
    """
    def __init__(self):
        self.colors = {
            'Red Bull Racing': '#1E41FF',
            'Ferrari': '#DC143C',
            'Mercedes': '#00D2BE',
            'McLaren': '#FF8700',
            'Aston Martin': '#006F62',
            'Alpine': '#0090FF',
            'Williams': '#005AFF',
            'AlphaTauri': '#2B4562',
            'Alfa Romeo': '#900000',
            'Haas': '#FFFFFF'
        }
    
    def plot_race_evolution(self, positions_df, session_info):
        """
        Create an interactive race evolution chart
        """
        if positions_df.empty:
            # Create sample race evolution
            laps = list(range(1, 51))
            drivers = ['Verstappen', 'Hamilton', 'Leclerc', 'Russell', 'Sainz']
            
            fig = go.Figure()
            
            for i, driver in enumerate(drivers):
                # Simulate position changes
                positions = [i + 1]
                for lap in range(1, 50):
                    change = np.random.randint(-1, 2)
                    new_pos = max(1, min(20, positions[-1] + change))
                    positions.append(new_pos)
                
                fig.add_trace(go.Scatter(
                    x=laps,
                    y=positions,
                    mode='lines+markers',
                    name=driver,
                    line=dict(width=3),
                    marker=dict(size=4)
                ))
        else:
            # Use real position data
            fig = go.Figure()
            
            for driver in positions_df['driver_number'].unique():
                driver_data = positions_df[positions_df['driver_number'] == driver]
                if not driver_data.empty:
                    fig.add_trace(go.Scatter(
                        x=driver_data.get('lap_number', range(len(driver_data))),
                        y=driver_data['position'],
                        mode='lines+markers',
                        name=f"Driver {driver}",
                        line=dict(width=3),
                        marker=dict(size=4)
                    ))
        
        fig.update_layout(
            title=f"Race Evolution - {session_info.get('meeting_name', 'F1 Race')}",
            xaxis_title="Lap Number",
            yaxis_title="Position",
            yaxis=dict(autorange="reversed", dtick=1),
            height=600,
            template="plotly_white",
            hovermode='x unified'
        )
        
        return fig
    
    def plot_driver_standings(self, standings_df):
        """
        Create driver championship standings visualization
        """
        if standings_df.empty:
            # Create sample standings
            drivers = ['Verstappen', 'Hamilton', 'Leclerc', 'Russell', 'Sainz', 
                      'Norris', 'Piastri', 'Alonso', 'Perez', 'Bottas']
            points = [450, 380, 320, 280, 250, 200, 180, 160, 140, 120]
            teams = ['Red Bull Racing', 'Mercedes', 'Ferrari', 'Mercedes', 'Ferrari',
                    'McLaren', 'McLaren', 'Aston Martin', 'Red Bull Racing', 'Alfa Romeo']
            
            standings_df = pd.DataFrame({
                'driver': drivers,
                'points': points,
                'team': teams
            })
        
        # Create bar chart
        fig = go.Figure()
        
        colors = [self.colors.get(team, '#666666') for team in standings_df['team']]
        
        fig.add_trace(go.Bar(
            x=standings_df['driver'],
            y=standings_df['points'],
            marker_color=colors,
            text=standings_df['points'],
            textposition='outside',
            name='Points'
        ))
        
        fig.update_layout(
            title="Driver Championship Standings",
            xaxis_title="Driver",
            yaxis_title="Points",
            height=500,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def plot_constructor_standings(self, constructor_standings):
        """
        Create constructor championship standings
        """
        if not constructor_standings:
            # Sample constructor standings
            teams = ['Red Bull Racing', 'Ferrari', 'Mercedes', 'McLaren', 'Aston Martin']
            points = [650, 520, 480, 350, 280]
            constructor_standings = pd.DataFrame({'team': teams, 'points': points})
        
        fig = go.Figure()
        
        colors = [self.colors.get(team, '#666666') for team in constructor_standings['team']]
        
        fig.add_trace(go.Bar(
            x=constructor_standings['team'],
            y=constructor_standings['points'],
            marker_color=colors,
            text=constructor_standings['points'],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Constructor Championship Standings",
            xaxis_title="Team",
            yaxis_title="Points",
            height=500,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def plot_pit_stop_analysis(self, pit_stops_df, session_info):
        """
        Visualize pit stop strategies and performance
        """
        if pit_stops_df.empty:
            # Create sample pit stop data
            drivers = ['Verstappen', 'Hamilton', 'Leclerc', 'Russell', 'Sainz']
            pit_data = []
            
            for driver in drivers:
                n_stops = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
                for stop in range(n_stops):
                    pit_data.append({
                        'driver': driver,
                        'lap_number': np.random.randint(15, 45),
                        'pit_duration': np.random.normal(25, 3),
                        'stop_number': stop + 1
                    })
            
            pit_stops_df = pd.DataFrame(pit_data)
        
        # Create subplot with pit stop timing and duration
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Pit Stop Timing', 'Pit Stop Duration'),
            vertical_spacing=0.15
        )
        
        # Pit stop timing
        for driver in pit_stops_df['driver'].unique():
            driver_pits = pit_stops_df[pit_stops_df['driver'] == driver]
            fig.add_trace(
                go.Scatter(
                    x=driver_pits['lap_number'],
                    y=[driver] * len(driver_pits),
                    mode='markers',
                    marker=dict(size=12, symbol='diamond'),
                    name=f"{driver} Pit Stops",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Pit stop duration
        fig.add_trace(
            go.Box(
                x=pit_stops_df['driver'],
                y=pit_stops_df['pit_duration'],
                name='Pit Duration',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"Pit Stop Analysis - {session_info.get('meeting_name', 'F1 Race')}",
            height=700,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Lap Number", row=1, col=1)
        fig.update_xaxes(title_text="Driver", row=2, col=1)
        fig.update_yaxes(title_text="Driver", row=1, col=1)
        fig.update_yaxes(title_text="Duration (seconds)", row=2, col=1)
        
        return fig
    
    def plot_performance_comparison(self, driver_stats, top_n=10):
        """
        Create comprehensive driver performance comparison
        """
        if not driver_stats:
            return None
        
        # Convert driver stats to DataFrame
        df = pd.DataFrame.from_dict(driver_stats, orient='index').reset_index()
        df.columns = ['driver'] + list(df.columns[1:])
        df = df.head(top_n)
        
        # Create radar chart for top drivers
        categories = ['Avg Finish', 'Win Rate', 'Podium Rate', 'Consistency']
        
        fig = go.Figure()
        
        for i, driver in enumerate(df['driver'][:5]):  # Top 5 drivers
            driver_data = df[df['driver'] == driver].iloc[0]
            
            values = [
                (21 - driver_data.get('avg_finish', 10)) / 20,  # Normalize
                driver_data.get('win_rate', 0),
                driver_data.get('podium_rate', 0),
                1 / (1 + driver_data.get('finish_std', 5))  # Consistency (lower std = higher consistency)
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=driver,
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Driver Performance Comparison (Top 5)",
            height=500
        )
        
        return fig

class F1RacePredictorAdvanced:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.trained = False
        self.data_collector = OpenF1DataCollector()
        self.pit_predictor = PitStopPredictor()
        self.visualizer = F1VisualizationEngine()
        self.driver_stats = {}
        self.team_stats = {}
        self.championship_standings = {}
        
    def collect_comprehensive_data(self, years=[2023, 2024], limit_races=None):
        """
        Collect comprehensive F1 data including pit stops and positions
        """
        print("Collecting comprehensive F1 data from OpenF1 API...")
        all_race_data = []
        all_pit_data = []
        all_position_data = []
        
        for year in years:
            print(f"Fetching data for {year}...")
            
            sessions = self.data_collector.get_sessions(year=year, session_type="Race")
            if sessions.empty:
                continue
            
            drivers = self.data_collector.get_drivers(year=year)
            drivers_dict = drivers.set_index('driver_number')['full_name'].to_dict() if not drivers.empty else {}
            
            race_count = 0
            for _, session in sessions.iterrows():
                if limit_races and race_count >= limit_races:
                    break
                    
                session_key = session['session_key']
                print(f"  Processing {session['meeting_name']} - {session['session_name']}")
                
                # Get race results
                results = self.data_collector.get_results(session_key)
                if results.empty:
                    continue
                
                # Get pit stop data
                pit_stops = self.data_collector.get_pit_stops(session_key)
                if not pit_stops.empty:
                    pit_stops['session_key'] = session_key
                    pit_stops['meeting_name'] = session['meeting_name']
                    all_pit_data.append(pit_stops)
                
                # Get position data
                positions = self.data_collector.get_positions(session_key)
                if not positions.empty:
                    positions['session_key'] = session_key
                    positions['meeting_name'] = session['meeting_name']
                    all_position_data.append(positions)
                
                # Get weather and qualifying data (existing logic)
                weather = self.data_collector.get_weather(session_key)
                weather_avg = self.get_average_weather(weather) if not weather.empty else {}
                
                qualifying_positions = self.get_qualifying_positions(year, session)
                
                # Process race results
                for _, result in results.iterrows():
                    if pd.isna(result['position']) or result['position'] == 0:
                        continue
                    
                    driver_number = result['driver_number']
                    driver_name = drivers_dict.get(driver_number, f"Driver_{driver_number}")
                    
                    # Calculate pit stop statistics for this driver
                    driver_pits = pit_stops[pit_stops['driver_number'] == driver_number] if not pit_stops.empty else pd.DataFrame()
                    pit_stats = self.calculate_pit_stats(driver_pits)
                    
                    race_entry = {
                        'year': year,
                        'session_key': session_key,
                        'meeting_name': session['meeting_name'],
                        'circuit_short_name': session['circuit_short_name'],
                        'driver_number': driver_number,
                        'driver_name': driver_name,
                        'team_name': result.get('team_name', 'Unknown'),
                        'qualifying_position': qualifying_positions.get(driver_number, 20),
                        'finishing_position': result['position'],
                        'points': result.get('points', 0),
                        'grid_position': result.get('grid_position', qualifying_positions.get(driver_number, 20)),
                        **weather_avg,
                        **pit_stats
                    }
                    
                    all_race_data.append(race_entry)
                
                race_count += 1
                time.sleep(0.1)
        
        # Combine all data
        race_df = pd.DataFrame(all_race_data) if all_race_data else pd.DataFrame()
        pit_df = pd.concat(all_pit_data, ignore_index=True) if all_pit_data else pd.DataFrame()
        position_df = pd.concat(all_position_data, ignore_index=True) if all_position_data else pd.DataFrame()
        
        if race_df.empty:
            print("No race data collected. Using sample data instead.")
            return self.create_sample_comprehensive_data()
        
        print(f"Collected {len(race_df)} race entries, {len(pit_df)} pit stops, {len(position_df)} position records")
        
        # Calculate comprehensive statistics
        self.calculate_comprehensive_stats(race_df)
        
        return {
            'races': race_df,
            'pit_stops': pit_df,
            'positions': position_df
        }
    
    def calculate_pit_stats(self, pit_stops_df):
        """Calculate pit stop statistics for a driver in a race"""
        if pit_stops_df.empty:
            return {
                'total_pit_stops': 0,
                'avg_pit_duration': 25.0,
                'fastest_pit': 25.0,
                'total_pit_time': 0.0,
                'pit_strategy': 'unknown'
            }
        
        return {
            'total_pit_stops': len(pit_stops_df),
            'avg_pit_duration': pit_stops_df['pit_duration'].mean() if 'pit_duration' in pit_stops_df.columns else 25.0,
            'fastest_pit': pit_stops_df['pit_duration'].min() if 'pit_duration' in pit_stops_df.columns else 25.0,
            'total_pit_time': pit_stops_df['pit_duration'].sum() if 'pit_duration' in pit_stops_df.columns else 0.0,
            'pit_strategy': f"{len(pit_stops_df)}-stop"
        }
    
    def get_qualifying_positions(self, year, session):
        """Get qualifying positions for a race session"""
        all_sessions = self.data_collector.get_sessions(year=year)
        if all_sessions.empty:
            return {}
        
        qualifying_session = all_sessions[
            (all_sessions['meeting_key'] == session['meeting_key']) & 
            (all_sessions['session_type'] == 'Qualifying')
        ]
        
        qualifying_positions = {}
        if not qualifying_session.empty:
            qual_key = qualifying_session.iloc[0]['session_key']
            qual_results = self.data_collector.get_results(qual_key)
            if not qual_results.empty:
                qualifying_positions = qual_results.set_index('driver_number')['position'].to_dict()
        
        return qualifying_positions
    
    def get_average_weather(self, weather_df):
        """Calculate average weather conditions"""
        if weather_df.empty:
            return {
                'air_temperature': 25.0,
                'track_temperature': 35.0,
                'humidity': 60.0,
                'wind_speed': 5.0,
                'rainfall': 0
            }
        
        return {
            'air_temperature': weather_df['air_temperature'].mean() if 'air_temperature' in weather_df.columns else 25.0,
            'track_temperature': weather_df['track_temperature'].mean() if 'track_temperature' in weather_df.columns else 35.0,
            'humidity': weather_df['humidity'].mean() if 'humidity' in weather_df.columns else 60.0,
            'wind_speed': weather_df['wind_speed'].mean() if 'wind_speed' in weather_df.columns else 5.0,
            'rainfall': weather_df['rainfall'].sum() if 'rainfall' in weather_df.columns else 0
        }
    
    def calculate_comprehensive_stats(self, df):
        """Calculate comprehensive driver and team statistics"""
        # Driver statistics including pit stop performance
        driver_stats = df.groupby('driver_name').agg({
            'finishing_position': ['mean', 'std', 'count'],
            'points': ['sum', 'mean'],
            'qualifying_position': 'mean',
            'total_pit_stops': 'mean',
            'avg_pit_duration': 'mean',
            'total_pit_time': 'mean'
        }).round(3)
        
        driver_stats.columns = ['avg_finish', 'finish_std', 'races_count', 
                               'total_points', 'avg_points', 'avg_qualifying',
                               'avg_pit_stops', 'avg_pit_duration', 'avg_pit_time']
        
        # Calculate win rate and podium rate
        wins = df[df['finishing_position'] == 1].groupby('driver_name').size()
        podiums = df[df['finishing_position'] <= 3].groupby('driver_name').size()
        
        driver_stats['win_rate'] = (wins / driver_stats['races_count']).fillna(0)
        driver_stats['podium_rate'] = (podiums / driver_stats['races_count']).fillna(0)
        
        self.driver_stats = driver_stats.to_dict('index')
        
        # Team statistics
        team_stats = df.groupby('team_name').agg({
            'finishing_position': ['mean', 'std'],
            'points': ['sum', 'mean'],
            'qualifying_position': 'mean',
            'total_pit_stops': 'mean',
            'avg_pit_duration': 'mean'
        }).round(3)
        
        team_stats.columns = ['avg_finish', 'finish_std', 'total_points', 'avg_points', 
                             'avg_qualifying', 'avg_pit_stops', 'avg_pit_duration']
        self.team_stats = team_stats.to_dict('index')
        
        # Championship standings
        current_standings = df.groupby('driver_name')['points'].sum().sort_values(ascending=False)
        constructor_standings = df.groupby('team_name')['points'].sum().sort_values(ascending=False)
        
        self.championship_standings = {
            'drivers': current_standings.to_dict(),
            'constructors': constructor_standings.to_dict()
        }
    
    def create_sample_comprehensive_data(self):
        """Create sample data with pit stops and positions"""
        # Use existing sample data creation but add pit stop data
        races_df = self.create_sample_data(100)
        
        # Create sample pit stop data
        pit_data = []
        position_data = []
        
        for session_key in races_df['session_key'].unique():
            session_races = races_df[races_df['session_key'] == session_key]
            
            for _, race in session_races.iterrows():
                # Simulate pit stops
                n_stops = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
                for stop in range(n_stops):
                    pit_data.append({
                        'session_key': session_key,
                        'driver_number': race['driver_number'],
                        'lap_number': np.random.randint(10, 50),
                        'pit_duration': np.random.normal(25, 3),
                        'stop_number': stop + 1
                    })
                
                # Simulate position changes throughout race
                for lap in range(1, 51):
                    position_data.append({
                        'session_key': session_key,
                        'driver_number': race['driver_number'],
                        'lap_number': lap,
                        'position': max(1, min(20, race['finishing_position'] + np.random.randint(-2, 3)))
                    })
        
        pit_df = pd.DataFrame(pit_data)
        position_df = pd.DataFrame(position_data)
        
        return {
            'races': races_df,
            'pit_stops': pit_df,
            'positions': position_df
        }
    
    def create_sample_data(self, n_races=500):
        """Create sample race data"""
        np.random.seed(42)
        
        circuits = ['Monaco', 'Silverstone', 'Monza', 'Spa', 'Singapore', 'Suzuka', 
                   'Interlagos', 'Austin', 'Barcelona', 'Hungaroring']
        
        drivers = ['Hamilton', 'Verstappen', 'Leclerc', 'Russell', 'Sainz', 'Norris',
                  'Piastri', 'Alonso', 'Perez', 'Bottas', 'Gasly', 'Ocon', 'Albon',
                  'Stroll', 'Tsunoda', 'Ricciardo', 'Hulkenberg', 'Magnussen', 'Zhou', 'Sargeant']
        
        teams = ['Mercedes', 'Red Bull Racing', 'Ferrari', 'McLaren', 'Aston Martin', 
                'Alpine', 'Williams', 'AlphaTauri', 'Haas', 'Alfa Romeo']
        
        data = []
        
        for race_id in range(n_races):
            circuit = np.random.choice(circuits)
            race_drivers = np.random.choice(drivers, size=20, replace=False)
            race_teams = np.random.choice(teams, size=20, replace=True)
            
            for i, driver in enumerate(race_drivers):
                qualifying_pos = i + 1 + np.random.randint(-2, 3)
                qualifying_pos = max(1, min(20, qualifying_pos))
                
                # Simulate finishing position
                base_finish = qualifying_pos + np.random.normal(0, 3)
                finishing_position = max(1, min(20, round(base_finish)))
                
                points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
                points = points_map.get(finishing_position, 0)
                
                # Add pit stop statistics
                n_pit_stops = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
                avg_pit_duration = np.random.normal(25, 3)
                
                data.append({
                    'year': 2024,
                    'session_key': race_id,
                    'meeting_name': f'Race_{race_id}',
                    'circuit_short_name': circuit,
                    'driver_number': i + 1,
                    'driver_name': driver,
                    'team_name': race_teams[i],
                    'qualifying_position': qualifying_pos,
                    'finishing_position': finishing_position,
                    'points': points,
                    'grid_position': qualifying_pos,
                    'air_temperature': np.random.normal(25, 8),
                    'track_temperature': np.random.normal(35, 10),
                    'humidity': np.random.uniform(30, 90),
                    'wind_speed': np.random.uniform(0, 20),
                    'rainfall': np.random.exponential(0.5) if np.random.random() > 0.8 else 0,
                    'total_pit_stops': n_pit_stops,
                    'avg_pit_duration': avg_pit_duration,
                    'fastest_pit': avg_pit_duration - np.random.uniform(0, 3),
                    'total_pit_time': n_pit_stops * avg_pit_duration,
                    'pit_strategy': f"{n_pit_stops}-stop"
                })
        
        return pd.DataFrame(data)
    
    def prepare_enhanced_features(self, df):
        """
        Prepare enhanced features including pit stop data
        """
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['driver_name', 'team_name', 'circuit_short_name', 'pit_strategy']
        
        for col in categorical_columns:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col + '_encoded'] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    known_categories = set(self.label_encoders[col].classes_)
                    df_processed[col + '_encoded'] = df_processed[col].astype(str).apply(
                        lambda x: self.label_encoders[col].transform([x])[0] if x in known_categories else -1
                    )
        
        # Add historical performance features
        df_processed['driver_avg_finish'] = df_processed['driver_name'].map(
            lambda x: self.driver_stats.get(x, {}).get('avg_finish', 10.0)
        )
        df_processed['driver_win_rate'] = df_processed['driver_name'].map(
            lambda x: self.driver_stats.get(x, {}).get('win_rate', 0.0)
        )
        df_processed['driver_podium_rate'] = df_processed['driver_name'].map(
            lambda x: self.driver_stats.get(x, {}).get('podium_rate', 0.0)
        )
        df_processed['driver_avg_pit_time'] = df_processed['driver_name'].map(
            lambda x: self.driver_stats.get(x, {}).get('avg_pit_duration', 25.0)
        )
        
        # Add team performance features
        df_processed['team_avg_finish'] = df_processed['team_name'].map(
            lambda x: self.team_stats.get(x, {}).get('avg_finish', 10.0)
        )
        df_processed['team_avg_points'] = df_processed['team_name'].map(
            lambda x: self.team_stats.get(x, {}).get('avg_points', 5.0)
        )
        df_processed['team_avg_pit_time'] = df_processed['team_name'].map(
            lambda x: self.team_stats.get(x, {}).get('avg_pit_duration', 25.0)
        )
        
        # Create enhanced engineered features
        df_processed['qualifying_advantage'] = 21 - df_processed['qualifying_position']
        df_processed['grid_penalty'] = (df_processed['grid_position'] - df_processed['qualifying_position']).fillna(0)
        df_processed['temperature_diff'] = df_processed['track_temperature'] - df_processed['air_temperature']
        df_processed['is_wet'] = (df_processed['rainfall'] > 0).astype(int)
        df_processed['weather_difficulty'] = df_processed['rainfall'] * df_processed['wind_speed']
        
        # Pit stop strategy features
        df_processed['pit_efficiency'] = df_processed['total_pit_time'] / df_processed['total_pit_stops']
        df_processed['pit_stop_advantage'] = 25.0 - df_processed.get('avg_pit_duration', 25.0)  # Relative to average
        df_processed['aggressive_strategy'] = (df_processed['total_pit_stops'] >= 3).astype(int)
        
        # Driver-team-strategy synergy
        df_processed['driver_team_performance'] = (
            (21 - df_processed['driver_avg_finish']) * (21 - df_processed['team_avg_finish'])
        ) / 400
        
        df_processed['pit_strategy_efficiency'] = (
            df_processed['pit_stop_advantage'] * df_processed['driver_team_performance']
        )
        
        # Select enhanced features for modeling
        feature_columns = [
            'qualifying_position', 'qualifying_advantage', 'grid_penalty',
            'air_temperature', 'track_temperature', 'temperature_diff',
            'humidity', 'wind_speed', 'rainfall', 'is_wet', 'weather_difficulty',
            'driver_avg_finish', 'driver_win_rate', 'driver_podium_rate', 'driver_avg_pit_time',
            'team_avg_finish', 'team_avg_points', 'team_avg_pit_time',
            'total_pit_stops', 'pit_efficiency', 'pit_stop_advantage', 'aggressive_strategy',
            'driver_team_performance', 'pit_strategy_efficiency',
            'driver_name_encoded', 'team_name_encoded', 'circuit_short_name_encoded'
        ]
        
        # Add pit strategy encoding if available
        if 'pit_strategy_encoded' in df_processed.columns:
            feature_columns.append('pit_strategy_encoded')
        
        # Fill missing values
        for col in feature_columns:
            if col in df_processed.columns:
                if df_processed[col].dtype in ['float64', 'int64']:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                else:
                    df_processed[col] = df_processed[col].fillna(0)
        
        self.feature_columns = feature_columns
        return df_processed[feature_columns]
    
    def train_enhanced_model(self, data_dict=None, target_column='finishing_position'):
        """
        Train the enhanced F1 prediction model with pit stop data
        """
        if data_dict is None:
            data_dict = self.collect_comprehensive_data(years=[2024], limit_races=5)
        
        df = data_dict['races']
        if df.empty:
            print("No training data available")
            return None
        
        print("Preparing enhanced features...")
        X = self.prepare_enhanced_features(df)
        y = df[target_column]
        
        # Remove rows with missing target values
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            print("No valid training data after preprocessing")
            return None
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Training enhanced model...")
        
        # Enhanced models with pit stop considerations
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=150, random_state=42, max_depth=12, min_samples_split=5),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=150, random_state=42, max_depth=8, learning_rate=0.1)
        }
        
        best_score = float('inf')
        best_model = None
        best_name = None
        
        for name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"{name} - MAE: {mae:.3f}")
            
            if mae < best_score:
                best_score = mae
                best_model = pipeline
                best_name = name
        
        self.model = best_model
        self.trained = True
        
        # Enhanced evaluation
        print(f"\nBest Model: {best_name}")
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Enhanced Model Performance:")
        print(f"MAE: {mae:.3f} positions")
        print(f"RMSE: {np.sqrt(mse):.3f} positions")
        print(f"R²: {r2:.3f}")
        
        # Feature importance analysis
        if hasattr(self.model.named_steps['model'], 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.named_steps['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 15 Most Important Features:")
            print(importance_df.head(15))
            
            # Identify pit stop feature importance
            pit_features = importance_df[importance_df['feature'].str.contains('pit|strategy', case=False)]
            if not pit_features.empty:
                print(f"\nPit Stop Strategy Features Importance:")
                print(pit_features)
        
        return self.model
    
    def predict_race_with_strategy(self, race_data, include_pit_prediction=True):
        """
        Predict race results including pit stop strategies
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Predict finishing positions
        X = self.prepare_enhanced_features(race_data)
        predictions = self.model.predict(X)
        predictions = np.clip(np.round(predictions), 1, 20)
        
        results = race_data.copy()
        results['predicted_position'] = predictions.astype(int)
        
        # Calculate predicted points
        points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        results['predicted_points'] = results['predicted_position'].map(lambda x: points_map.get(x, 0))
        
        if include_pit_prediction:
            # Predict pit stop strategies
            race_distance = 50  # Assume 50 laps average
            weather_conditions = {
                'track_temperature': results['track_temperature'].mean(),
                'rainfall': results['rainfall'].mean(),
                'air_temperature': results['air_temperature'].mean()
            }
            
            strategies = self.pit_predictor.predict_pit_strategy(race_distance, weather_conditions)
            
            # Assign strategies based on predicted position (leaders get more aggressive)
            results = results.sort_values('predicted_position')
            for i, (idx, row) in enumerate(results.iterrows()):
                if i < 3:  # Top 3 get aggressive strategy
                    strategy_idx = min(2, len(strategies) - 1)
                elif i < 10:  # Middle pack gets balanced strategy
                    strategy_idx = 1 if len(strategies) > 1 else 0
                else:  # Back markers get conservative strategy
                    strategy_idx = 0
                
                if strategy_idx < len(strategies):
                    results.loc[idx, 'recommended_strategy'] = strategies[strategy_idx]['strategy']
                    results.loc[idx, 'predicted_pit_stops'] = len(strategies[strategy_idx]['pit_laps'])
                    results.loc[idx, 'strategy_risk'] = strategies[strategy_idx]['risk_level']
        
        return results.sort_values('predicted_position')
    
    def generate_comprehensive_report(self, data_dict):
        """
        Generate comprehensive race analysis with visualizations
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE F1 RACE ANALYSIS REPORT")
        print("=" * 80)
        
        races_df = data_dict['races']
        pit_stops_df = data_dict['pit_stops']
        positions_df = data_dict['positions']
        
        # Basic statistics
        print(f"\nDataset Overview:")
        print(f"Total Races Analyzed: {len(races_df['session_key'].unique())}")
        print(f"Total Race Entries: {len(races_df)}")
        print(f"Total Pit Stops: {len(pit_stops_df)}")
        print(f"Position Records: {len(positions_df)}")
        
        # Driver performance summary
        if self.driver_stats:
            print(f"\nTop 5 Drivers by Points:")
            driver_points = [(driver, stats['total_points']) for driver, stats in self.driver_stats.items()]
            driver_points.sort(key=lambda x: x[1], reverse=True)
            
            for i, (driver, points) in enumerate(driver_points[:5], 1):
                stats = self.driver_stats[driver]
                print(f"{i}. {driver}: {points:.0f} pts (Avg: P{stats['avg_finish']:.1f}, "
                      f"Wins: {stats['win_rate']*100:.1f}%, Podiums: {stats['podium_rate']*100:.1f}%)")
        
        # Championship standings
        if self.championship_standings:
            print(f"\nConstructor Championship:")
            for i, (team, points) in enumerate(list(self.championship_standings['constructors'].items())[:5], 1):
                print(f"{i}. {team}: {points:.0f} points")
        
        # Generate visualizations
        print(f"\nGenerating Visualizations...")
        
        # Sample session for visualization
        sample_session = races_df['session_key'].iloc[0] if not races_df.empty else None
        session_info = {'meeting_name': 'Sample F1 Race'}
        
        if sample_session:
            session_races = races_df[races_df['session_key'] == sample_session]
            session_info = {
                'meeting_name': session_races['meeting_name'].iloc[0],
                'circuit_short_name': session_races['circuit_short_name'].iloc[0]
            }
        
        # Create visualizations
        visualizations = {}
        
        try:
            # Race evolution
            session_positions = positions_df[positions_df['session_key'] == sample_session] if sample_session else pd.DataFrame()
            visualizations['race_evolution'] = self.visualizer.plot_race_evolution(session_positions, session_info)
            
            # Driver standings
            if self.championship_standings and 'drivers' in self.championship_standings:
                standings_data = pd.DataFrame([
                    {'driver': driver, 'points': points, 'team': 'Unknown'}
                    for driver, points in self.championship_standings['drivers'].items()
                ])
                visualizations['driver_standings'] = self.visualizer.plot_driver_standings(standings_data)
            
            # Constructor standings
            if self.championship_standings and 'constructors' in self.championship_standings:
                constructor_data = pd.DataFrame([
                    {'team': team, 'points': points}
                    for team, points in self.championship_standings['constructors'].items()
                ])
                visualizations['constructor_standings'] = self.visualizer.plot_constructor_standings(constructor_data)
            
            # Pit stop analysis
            session_pits = pit_stops_df[pit_stops_df['session_key'] == sample_session] if sample_session else pd.DataFrame()
            visualizations['pit_analysis'] = self.visualizer.plot_pit_stop_analysis(session_pits, session_info)
            
            # Performance comparison
            visualizations['performance_comparison'] = self.visualizer.plot_performance_comparison(self.driver_stats)
            
            print("✓ Race Evolution Chart")
            print("✓ Driver Championship Standings")
            print("✓ Constructor Championship Standings") 
            print("✓ Pit Stop Strategy Analysis")
            print("✓ Driver Performance Comparison")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
        
        return visualizations
    
    def save_visualizations(self, visualizations, output_dir="f1_analysis"):
        """
        Save all visualizations to HTML files
        """
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for name, fig in visualizations.items():
            if fig is not None:
                try:
                    filename = os.path.join(output_dir, f"{name}.html")
                    fig.write_html(filename)
                    print(f"Saved {name} to {filename}")
                except Exception as e:
                    print(f"Error saving {name}: {e}")

# Example usage and demonstration
if __name__ == "__main__":
    print("Advanced F1 Race Prediction System with Pit Stop Analysis")
    print("=" * 70)
    
    # Initialize the advanced predictor
    predictor = F1RacePredictorAdvanced()
    
    try:
        # Collect comprehensive data
        print("Collecting comprehensive F1 data...")
        data_dict = predictor.collect_comprehensive_data(years=[2024], limit_races=3)
        
        # Train the enhanced model
        print("\nTraining enhanced prediction model...")
        predictor.train_enhanced_model(data_dict)
        
        # Generate comprehensive analysis
        print("\nGenerating comprehensive analysis...")
        visualizations = predictor.generate_comprehensive_report(data_dict)
        
        # Show sample predictions with strategy
        sample_race = data_dict['races'][data_dict['races']['session_key'] == data_dict['races']['session_key'].iloc[0]].copy()
        if not sample_race.empty:
            print("\n" + "=" * 70)
            print("SAMPLE RACE PREDICTION WITH PIT STRATEGY")
            print("=" * 70)
            
            results = predictor.predict_race_with_strategy(sample_race)
            
            print(f"Race: {results.iloc[0]['meeting_name']}")
            print(f"Circuit: {results.iloc[0]['circuit_short_name']}")
            print("\nPredicted Results with Pit Strategy:")
            print("-" * 90)
            
            for i, row in results.head(10).iterrows():
                strategy = row.get('recommended_strategy', 'N/A')
                risk = row.get('strategy_risk', 'N/A')
                print(f"P{row['predicted_position']:2d}: {row['driver_name']:<20} ({row['team_name']:<15}) "
                      f"Q{row['qualifying_position']:2d} | {strategy} ({risk}) -> {row['predicted_points']:2d} pts")
            
            print(f"\nPredicted Podium with Strategies:")
            podium = results.head(3)
            for i, row in podium.iterrows():
                position_names = {1: "🥇 Winner", 2: "🥈 Second", 3: "🥉 Third"}
                strategy = row.get('recommended_strategy', 'Unknown')
                print(f"{position_names[row['predicted_position']]}: {row['driver_name']} ({row['team_name']}) - {strategy} strategy")
        
        # Optionally save visualizations
        # predictor.save_visualizations(visualizations)
        print(f"\nAnalysis complete! Use the visualization objects to display charts.")
        
    except Exception as e:
        print(f"Error with OpenF1 API: {e}")
        print("Falling back to sample data for demonstration...")
        
        # Fallback demonstration with sample data
        sample_data_dict = predictor.create_sample_comprehensive_data()
        predictor.calculate_comprehensive_stats(sample_data_dict['races'])
        predictor.train_enhanced_model(sample_data_dict)
        
        visualizations = predictor.generate_comprehensive_report(sample_data_dict)
        
        # Sample prediction
        sample_race = sample_data_dict['races'][sample_data_dict['races']['session_key'] == 0].copy()
        if not sample_race.empty:
            results = predictor.predict_race_with_strategy(sample_race)
            print("\nSample Race Prediction with Pit Strategy:")
            print("-" * 70)
            for i, row in results.head(5).iterrows():
                strategy = row.get('recommended_strategy', 'N/A')
                print(f"P{row['predicted_position']:2d}: {row['driver_name']:<15} Q{row['qualifying_position']:2d} | {strategy}")
        
        print(f"\nSample analysis complete!") 