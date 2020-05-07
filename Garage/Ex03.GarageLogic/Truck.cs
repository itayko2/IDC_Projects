using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public abstract class Truck : Vehicle
    {
        private const int k_TruckWheelsNumber = 12;
        private const float k_TruckWheelMaxAirPressure = 26;
        private readonly float r_MaxLoad;
        private bool m_IsCarryingHazardousMaterials;

        public Truck(
            string i_OwnersName,
            string i_OwnersPhone,
            string i_Model,
            string i_RegistrationPlate,
            Energy i_EnergyType,
            float i_EnergyLevel,
            float i_CurrentAirPressure,
            string i_ManufacturerName,
            GarageENums.eVehicleType i_VehicleType,
            float i_MaxLoad,
            bool i_IsCarryingHazardousMaterials)
            : base(
                  i_OwnersName,
                  i_OwnersPhone,
                  i_Model,
                  i_RegistrationPlate,
                  i_EnergyType,
                  i_EnergyLevel,
                  k_TruckWheelsNumber,
                  i_ManufacturerName,
                  i_CurrentAirPressure,
                  k_TruckWheelMaxAirPressure,
                  i_VehicleType)
        {
            r_MaxLoad = i_MaxLoad;
            m_IsCarryingHazardousMaterials = i_IsCarryingHazardousMaterials;
        }

        public static new List<VehicleProperties> GetRequiredProperties()
        {
            List<VehicleProperties> properties = Vehicle.GetRequiredProperties();
            properties.Add(new VehicleProperties(typeof(float), "Maximum Load Weight"));
            properties.Add(new VehicleProperties(typeof(bool), "Carrying Hazardous Materials"));

            return properties;
        }

        public override string ToString()
        {
            string truckInfo = string.Format("Truck Max Load: {1}{0}Is Carrying Hazardous Materials: {2}{0}", Environment.NewLine, r_MaxLoad, m_IsCarryingHazardousMaterials);

            return base.ToString() + truckInfo;
        }

        public bool IsCarryingHazardousMaterials
        {
            get { return m_IsCarryingHazardousMaterials; }
        }

        public float MaxLoad
        {
            get { return r_MaxLoad; }
        }
    }
}
