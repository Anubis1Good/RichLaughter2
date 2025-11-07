from typing import Dict, List, Any, Optional, Union
import pandas as pd


class WSBase:
    """
    Универсальный базовый класс для торговых стратегий.
    Поддерживает работу с множеством инструментов, датафреймов и позиций.
    """
    def __init__(
        self,
        symbols: Union[str, List[str]] = "IMOEXF",
        timeframes: Union[str, List[str]] = "1m",
        initial_positions: Union[int, Dict[str, int]] = 0,
        parameters: Dict[str, Any] = None
    ):
        """
        Инициализация стратегии.
        
        Args:
            symbols: Символ(ы) для торговли
            timeframes: Таймфрейм(ы) для анализа
            initial_positions: Начальные позиции
            parameters: Параметры стратегии
        """
        # Конвертируем в списки для унификации
        self.symbols = [symbols] if isinstance(symbols, str) else symbols
        self.timeframes = [timeframes] if isinstance(timeframes, str) else timeframes
        
        # Инициализируем позиции
        self.positions = self._init_positions(initial_positions)
        self.target_positions = self.positions.copy()
        
        # Параметры стратегии
        self.parameters = parameters or {}
        
        # Хранилище данных и состояний
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.indicators: Dict[str, Dict[str, Any]] = {}
        self.states: Dict[str, Any] = {}
        
        # Средние цены для каждого инструмента
        self.middle_prices = {symbol: None for symbol in self.symbols}

    def _init_positions(self, initial_positions: Union[int, Dict[str, int]]) -> Dict[str, int]:
        """Инициализирует словарь позиций."""
        if isinstance(initial_positions, dict):
            return initial_positions.copy()
        else:
            return {symbol: initial_positions for symbol in self.symbols}
    
    def update_dataframe(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Обновляет датафрейм для конкретного символа и таймфрейма."""
        key = f"{symbol}_{timeframe}"
        self.dataframes[key] = df.copy()
        
    def get_dataframe(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Возвращает датафрейм для символа и таймфрейма."""
        key = f"{symbol}_{timeframe}"
        return self.dataframes.get(key)
    
    def update_position(self, symbol: str, position: int):
        """Обновляет позицию для символа."""
        if symbol in self.positions:
            self.positions[symbol] = position
    
    def update_middle_price(self, symbol: str, middle_price: float):
        """Обновляет среднюю цену для символа."""
        if symbol in self.middle_prices:
            self.middle_prices[symbol] = middle_price
    
    def preprocessing(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str,
        position: Optional[int] = None,
        middle_price: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Предобработка данных. Может быть переопределена в дочерних классах.
        """
        if position is not None:
            self.update_position(symbol, position)
        if middle_price is not None:
            self.update_middle_price(symbol, middle_price)
            
        return df.copy()
    
    def calculate_indicators(self, symbol: str, timeframe: str):
        """
        Расчет индикаторов. Должен быть реализован в дочерних классах.
        """
        key = f"{symbol}_{timeframe}"
        df = self.get_dataframe(symbol, timeframe)
        
        if df is not None:
            # Пример структуры для хранения индикаторов
            self.indicators[key] = {
                # Здесь будут рассчитываться индикаторы
                # Например: 'sma': df['close'].rolling(20).mean()
            }
    
    def generate_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Генерация торговых сигналов. Должен быть реализован в дочерних классах.
        
        Returns:
            Словарь с сигналами и целевыми позициями
        """
        # Базовая реализация - возвращает текущую позицию
        return {
            'symbol': symbol,
            'target_position': self.positions.get(symbol, 0),
            'signals': {},
            'timestamp': pd.Timestamp.now()
        }
    
    def portfolio_management(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """
        Управление портфелем. Может быть переопределено для арбитражных стратегий.
        
        Args:
            signals: Словарь сигналов для всех символов
            
        Returns:
            Словарь целевых позиций для всех символов
        """
        target_positions = {}
        
        for symbol, signal_data in signals.items():
            target_positions[symbol] = signal_data['target_position']
            
        return target_positions
    
    def risk_management(self, current_positions: Dict[str, int]) -> Dict[str, int]:
        """
        Управление рисками. Может быть переопределено в дочерних классах.
        """
        # Базовая реализация - просто возвращает текущие позиции
        return current_positions.copy()
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Основной метод выполнения стратегии.
        
        Returns:
            Словарь с целевыми позициями и дополнительной информацией
        """
        # 1. Сбор сигналов по всем символам
        all_signals = {}
        for symbol in self.symbols:
            signal = self.generate_signals(symbol)
            all_signals[symbol] = signal
        
        # 2. Управление портфелем
        target_positions = self.portfolio_management(all_signals)
        
        # 3. Управление рисками
        final_positions = self.risk_management(target_positions)
        
        # 4. Обновление целевых позиций
        self.target_positions.update(final_positions)
        
        return {
            'target_positions': final_positions,
            'signals': all_signals,
            'current_positions': self.positions.copy(),
            'middle_prices': self.middle_prices.copy()
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Возвращает текущее состояние стратегии."""
        return {
            'symbols': self.symbols.copy(),
            'timeframes': self.timeframes.copy(),
            'positions': self.positions.copy(),
            'target_positions': self.target_positions.copy(),
            'middle_prices': self.middle_prices.copy(),
            'parameters': self.parameters.copy(),
            'states': self.states.copy()
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Восстанавливает состояние стратегии."""
        self.symbols = state.get('symbols', self.symbols)
        self.timeframes = state.get('timeframes', self.timeframes)
        self.positions = state.get('positions', self.positions)
        self.target_positions = state.get('target_positions', self.target_positions)
        self.middle_prices = state.get('middle_prices', self.middle_prices)
        self.parameters = state.get('parameters', self.parameters)
        self.states = state.get('states', self.states)