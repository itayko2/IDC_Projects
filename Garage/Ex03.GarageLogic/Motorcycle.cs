using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public abstract class Motorcycle : Vehicle
    {
        private const int k_MotorcycleWheelsNumber = 2;
        private const float k_MotorcycleWheelMaxAirPressure = 33;
        private Motorcycle.eMotorcycleLicenseType m_LicenseType;
        private int m_EngineDisplacement;

        public Motorcycle(
            string i_OwnersName,
            string i_OwnersPhone,
            string i_Model,
            string i_RegistrationPlate,
            Energy i_EnergyType,
            float i_EnergyLevel,
            float i_CurrentAirPressure,
            string i_ManufacturerName,
            GarageENums.eVehicleType i_VehicleType,
            Motorcycle.eMotorcycleLicenseType i_LicenseType,
            int i_EngineDisplacement)
            : base(
                  i_OwnersName,
                  i_OwnersPhone,
                  i_Model,
                  i_RegistrationPlate,
                  i_EnergyType,
                  i_EnergyLevel,
                  k_MotorcycleWheelsNumber,
                  i_ManufacturerName,
                  i_CurrentAirPressure,
                  k_MotorcycleWheelMaxAirPressure,
                  i_VehicleType)
        {
            m_EngineDisplacement = i_EngineDisplacement;
            m_LicenseType = i_LicenseType;
        }

        public static new List<VehicleProperties> GetRequiredProperties()
        {
            List<VehicleProperties> properties = Vehicle.GetRequiredProperties();
            properties.Add(new VehicleProperties(typeof(Motorcycle.eMotorcycleLicenseType), "License Type"));
            properties.Add(new VehicleProperties(typeof(int), "Engine Displacement"));

            return properties;
        }

        public Motorcycle.eMotorcycleLicenseType LicenseType
        {
            get
            { return m_LicenseType; }
        }

        public int EngineDisplacement
        {
            get { return m_EngineDisplacement; }
        }

        public override string ToString()
        {
            string motorcycleInfo = string.Format("License Type: {1}{0}Engine Displacement: {2}{0}", Environment.NewLine, m_LicenseType, m_EngineDisplacement);

            return base.ToString() + motorcycleInfo;
        }

        public enum eMotorcycleLicenseType
        {
            A = 1,
            A1 = 2,
            A2 = 3,
            B = 4
        }
    }
}
