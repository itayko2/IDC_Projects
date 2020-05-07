using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public class ElectricMotorcycle : Motorcycle
    {
        private const float k_MaxChargeTime = 1.4f;
        private const GarageENums.eVehicleType k_VehicleType = GarageENums.eVehicleType.ElectricMotorcycle;

        public ElectricMotorcycle(
            string i_OwnersName,
            string i_OwnersPhone,
            string i_Model,
            string i_RegistrationPlate,
            float i_EnergyLevel,
            float i_CurrentAirPressure,
            string i_ManufacturerName,
            Motorcycle.eMotorcycleLicenseType i_LicenseType,
            int i_EngineDisplacement)
            : base(
                  i_OwnersName,
                  i_OwnersPhone,
                  i_Model,
                  i_RegistrationPlate,
                  new ElectricBattery(k_MaxChargeTime, i_EnergyLevel),
                  i_EnergyLevel,
                  i_CurrentAirPressure,
                  i_ManufacturerName,
                  k_VehicleType,
                  i_LicenseType,
                  i_EngineDisplacement)
        {
        }

        public float MaxChargeTime
        {
            get
            { return k_MaxChargeTime; }
        }
    }
}
