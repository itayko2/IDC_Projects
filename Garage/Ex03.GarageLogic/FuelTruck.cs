using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public class FuelTruck : Truck
    {
        private const FuelTank.eFuelType k_FuelType = FuelTank.eFuelType.Soler;
        private const float k_FuelTankCapacity = 110f;
        private const GarageENums.eVehicleType k_VehicleType = GarageENums.eVehicleType.FuelTruck;

        public FuelTruck(
            string i_OwnersName,
            string i_OwnersPhone,
            string i_Model,
            string i_RegistrationPlate,
            float i_EnergyLevel,
            float i_CurrentAirPressure,
            string i_ManufacturerName,
            float i_MaxLoad,
            bool i_IsCarryingHazardousMaterials)
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
                  i_MaxLoad,
                  i_IsCarryingHazardousMaterials)
        {
        }

        public float FuelTankCapacity
        {
            get { return k_FuelTankCapacity; }
        }

        public FuelTank.eFuelType FuelType
        {
            get { return k_FuelType; }
        }
    }
}
