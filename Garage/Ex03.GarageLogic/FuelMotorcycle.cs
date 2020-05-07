using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public class FuelMotorcycle : Motorcycle
    {
        private const FuelTank.eFuelType k_FuelType = FuelTank.eFuelType.Octan95;
        private const float k_FuelTankCapacity = 8f;
        private const GarageENums.eVehicleType k_VehicleType = GarageENums.eVehicleType.FuelMotorcycle;

        public FuelMotorcycle(
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
                  new FuelTank(k_FuelTankCapacity, i_EnergyLevel, k_FuelType),
                  i_EnergyLevel,
                  i_CurrentAirPressure,
                  i_ManufacturerName,
                  k_VehicleType,
                  i_LicenseType,
                  i_EngineDisplacement)
        {
        }

        public float FuelTankTankCapacity
        {
            get { return k_FuelTankCapacity; }
        }

        public FuelTank.eFuelType FuelTankType
        {
            get { return k_FuelType; }
        }
    }
}
