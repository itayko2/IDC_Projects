using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public abstract class Car : Vehicle
    {
        private const int k_CarWheelsNumber = 4;
        private const float k_CarWheelMaxPressure = 31;
        private Car.eCarColor m_CarColor;
        private Car.eNumberOfCarDoors m_NumberOfDoors;

        public Car(
            string i_OwnersName,
            string i_OwnersPhone,
            string i_Model,
            string i_RegistrationPlate,
            Energy i_EnergyType,
            float i_EnergyLevel,
            float i_CurrentAirPressure,
            string i_ManufacturerName,
            GarageENums.eVehicleType i_VehicleType,
            Car.eCarColor i_CarColor,
            Car.eNumberOfCarDoors i_NumberOfDoors)
            : base(
                    i_OwnersName,
                    i_OwnersPhone,
                    i_Model,
                    i_RegistrationPlate,
                    i_EnergyType,
                    i_EnergyLevel,
                    k_CarWheelsNumber,
                    i_ManufacturerName,
                    i_CurrentAirPressure,
                    k_CarWheelMaxPressure,
                    i_VehicleType)
        {
            m_CarColor = i_CarColor;
            m_NumberOfDoors = i_NumberOfDoors;
        }

        public static new List<VehicleProperties> GetRequiredProperties()
        {
            List<VehicleProperties> properties = Vehicle.GetRequiredProperties();
            properties.Add(new VehicleProperties(typeof(Car.eCarColor), "Color Of Car"));
            properties.Add(new VehicleProperties(typeof(Car.eNumberOfCarDoors), "Number Of Doors"));

            return properties;
        }

        public override string ToString()
        {
            string carInfo = string.Format("Car Color: {1}{0}Number Of Doors: {2}{0}", Environment.NewLine, m_CarColor, m_NumberOfDoors);

            return base.ToString() + carInfo;
        }

        public enum eCarColor
        {
            Red = 1,
            Blue = 2,
            Black = 3,
            Grey = 4
        }

        public enum eNumberOfCarDoors
        {
            Two = 2,
            Three = 3,
            Four = 4,
            Five = 5
        }
    }
}
