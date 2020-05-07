using System;
namespace Ex03.GarageLogic
{
    internal static class Validation
    {

        public static float FloatValueInRange(float i_MinIndex, float i_MaxIndex, string i_Input)
        {
            float returnValue = -1;
            if (!(float.TryParse(i_Input, out returnValue) && returnValue >= i_MinIndex && returnValue <= i_MaxIndex))
            {
                throw new FormatException("Invalid input. Please try again with a valid value.");
            }

            return returnValue;
        }

        public static float GetFloatValueInRange(float i_MinIndex, float i_MaxIndex)
        {
            float returnValue = -1;
            string input = Console.ReadLine();
            try
            {
                returnValue = Ex03.GarageLogic.Validation.FloatValueInRange(i_MinIndex, i_MaxIndex, input);
            }
            catch (Exception)
            {
                Console.WriteLine("Invalid input.Please try again with a valid value.");
                returnValue = GetFloatValueInRange(i_MinIndex, i_MaxIndex);
            }

            return returnValue;
        }

        public static float GetCurrentAirPressure(int i_TypeOfCar, Garage s_Garage)
        {
            float maxAirPressure = 0;
            switch (i_TypeOfCar)
            {
                case 1:
                case 2:
                    maxAirPressure = s_Garage.MaxAirPressureCar;
                    break;
                case 3:
                case 4:
                    maxAirPressure = s_Garage.MaxAirPressureMotorcycle;
                    break;
                case 5:
                    maxAirPressure = s_Garage.MaxAirPressureTruck;
                    break;
            }

            return GetFloatValueInRange(0, maxAirPressure);
        }

        public static float GetEnergyLevel(int i_TypeOfCar, Garage s_Garage)
        {
            float energyLevel = 0;
            switch (i_TypeOfCar)
            {
                case 1:
                    energyLevel = s_Garage.TankSizeFuelCar;
                    break;
                case 2:
                    energyLevel = s_Garage.BatteryTimeElectricityCar;
                    break;
                case 3:
                    energyLevel = s_Garage.TankSizeFuelMotorcycle;
                    break;
                case 4:
                    energyLevel = s_Garage.BatteryTimeElectricityMotorcycle;
                    break;
                case 5:
                    energyLevel = s_Garage.TankSizeFuelTruck;
                    break;
            }
            return GetFloatValueInRange(0, energyLevel);
        }
    }
}

