"""
Weather Service module - provides weather forecasts for Tenerife.
Supports simulated and real API implementations.
"""

import json
import random
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from src.logger import logger


# --- Pydantic Schemas ---

class WeatherRequest(BaseModel):
    """Schema for weather request."""
    date: str = Field(..., description="Fecha en formato YYYY-MM-DD")
    location: str = Field(default="Tenerife", description="Ubicación (default: Tenerife)")
    
    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Formato de fecha inválido. Usar YYYY-MM-DD")
        return v


class WeatherResponse(BaseModel):
    """Schema for weather response."""
    date: str
    location: str
    temperature_high: int
    temperature_low: int
    condition: str
    humidity: int
    wind_speed: int
    recommendation: str
    simulated: bool = True


# --- OpenAI Tool Schema ---

WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Obtiene el pronóstico del tiempo para una fecha específica en Tenerife. Usar cuando el usuario pregunte sobre el clima, tiempo, temperatura o condiciones meteorológicas.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Fecha para el pronóstico en formato YYYY-MM-DD"
                },
                "location": {
                    "type": "string",
                    "description": "Ubicación en Tenerife (opcional, default: Tenerife general)"
                }
            },
            "required": ["date"]
        }
    }
}


# --- Weather Service Class ---

class WeatherService:
    """Service for getting weather forecasts."""
    
    # Typical Tenerife weather conditions (weighted for realism)
    CONDITIONS = [
        ("Soleado", 40),
        ("Parcialmente nublado", 30),
        ("Nublado", 15),
        ("Lluvia ligera", 10),
        ("Ventoso", 5)
    ]
    
    # Recommendations based on conditions
    RECOMMENDATIONS = {
        "Soleado": "Perfecto para la playa. No olvides protector solar y agua.",
        "Parcialmente nublado": "Buen día para turismo. Lleva gafas de sol por si acaso.",
        "Nublado": "Ideal para visitar museos o el casco histórico de La Laguna.",
        "Lluvia ligera": "Lleva un paraguas. Buen día para visitar centros comerciales o restaurantes.",
        "Ventoso": "Cuidado en zonas de acantilados. Evita actividades acuáticas."
    }
    
    def __init__(self, simulated: bool = True, api_key: Optional[str] = None):
        """
        Initialize weather service.
        
        Args:
            simulated: If True, use simulated data. If False, use real API.
            api_key: API key for real weather service (required if simulated=False)
        """
        self.simulated = simulated
        self.api_key = api_key
        
        if not simulated and not api_key:
            raise ValueError("API key required for real weather service")
        
        logger.info(f"WeatherService initialized (simulated={simulated})")
    
    def _get_simulated_weather(self, request: WeatherRequest) -> WeatherResponse:
        """Generate realistic simulated weather for Tenerife."""
        
        # Parse date to adjust temperature by season
        date_obj = datetime.strptime(request.date, "%Y-%m-%d")
        month = date_obj.month
        
        # Tenerife temperature ranges by season (mild year-round)
        if month in [12, 1, 2]:  # Winter
            temp_high = random.randint(18, 22)
            temp_low = random.randint(14, 17)
        elif month in [3, 4, 5]:  # Spring
            temp_high = random.randint(20, 24)
            temp_low = random.randint(15, 18)
        elif month in [6, 7, 8]:  # Summer
            temp_high = random.randint(25, 30)
            temp_low = random.randint(19, 23)
        else:  # Fall
            temp_high = random.randint(22, 26)
            temp_low = random.randint(17, 20)
        
        # Select condition based on weights
        conditions, weights = zip(*self.CONDITIONS)
        condition = random.choices(conditions, weights=weights, k=1)[0]
        
        return WeatherResponse(
            date=request.date,
            location=request.location,
            temperature_high=temp_high,
            temperature_low=temp_low,
            condition=condition,
            humidity=random.randint(50, 75),
            wind_speed=random.randint(5, 25),
            recommendation=self.RECOMMENDATIONS[condition],
            simulated=True
        )
    
    def _get_real_weather(self, request: WeatherRequest) -> WeatherResponse:
        """
        Get real weather from API.
        
        TODO: Implement with real weather API (e.g., OpenWeatherMap, WeatherAPI)
        """
        # Placeholder for real implementation
        # Example with OpenWeatherMap:
        # import requests
        # url = f"https://api.openweathermap.org/data/2.5/forecast?q={request.location}&appid={self.api_key}"
        # response = requests.get(url)
        # data = response.json()
        # ... parse and return WeatherResponse
        
        raise NotImplementedError("Real weather API not implemented yet")
    
    def get_weather(self, date: str, location: str = "Tenerife") -> dict:
        """
        Get weather forecast.
        
        Args:
            date: Date in YYYY-MM-DD format
            location: Location in Tenerife
            
        Returns:
            Weather data as dictionary
        """
        logger.info(f"Weather request: date={date}, location={location}, simulated={self.simulated}")
        
        try:
            # Validate request
            request = WeatherRequest(date=date, location=location)
            
            # Check date is not too far in future (max 7 days for realism)
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            today = datetime.now()
            days_diff = (date_obj - today).days
            
            if days_diff > 7:
                logger.warning(f"Date too far in future: {date}")
                return {
                    "error": True,
                    "message": "Solo puedo proporcionar pronósticos hasta 7 días en el futuro."
                }
            
            if days_diff < -1:
                logger.warning(f"Date in the past: {date}")
                return {
                    "error": True,
                    "message": "No puedo proporcionar el tiempo para fechas pasadas."
                }
            
            # Get weather
            if self.simulated:
                response = self._get_simulated_weather(request)
            else:
                response = self._get_real_weather(request)
            
            logger.info(f"Weather response: {response.condition}, {response.temperature_high}°C")
            return response.model_dump()
            
        except ValueError as e:
            logger.error(f"Weather request validation error: {e}")
            return {
                "error": True,
                "message": str(e)
            }
        except Exception as e:
            logger.error(f"Weather service error: {e}")
            return {
                "error": True,
                "message": "Error al obtener el pronóstico del tiempo."
            }
    
    def get_tool_schema(self) -> dict:
        """Return the OpenAI tool schema for this function."""
        return WEATHER_TOOL_SCHEMA
    
    def parse_tool_call(self, tool_call) -> dict:
        """Parse OpenAI tool call and execute the function."""
        try:
            arguments = json.loads(tool_call.function.arguments)
            date = arguments.get("date")
            location = arguments.get("location", "Tenerife")
            
            return self.get_weather(date=date, location=location)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool call arguments: {e}")
            return {
                "error": True,
                "message": "Error al procesar la solicitud del tiempo."
            }