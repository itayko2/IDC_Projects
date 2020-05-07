using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public abstract class Vehicle
    {
        private readonly string r_Model;
        private readonly string r_RegistrationPlate;
        private string m_OwnersName;
        private string m_OwnersPhone;
        private List<Wheel> m_Wheels;
        private Energy m_EnergyType;
        private float m_EnergyLevel;
        private float m_CurrentAirPressure;
        private int m_NumberOfWheels;
        private GarageENums.eVehicleType m_VehicleType;

        public Vehicle(

            string i_OwnersName,
            string i_OwnersPhone,
            string i_Model,
            string i_RegistrationPlate,
            Energy i_EnergyType,
            float i_EnergyLevel,
            int i_NumberOfWheels,
            string i_ManufacturerName,
            float i_CurrentAirPressure,
            float i_MaxAirPressure,
            GarageENums.eVehicleType i_VehicleType)
        {
            m_OwnersName = i_OwnersName;
            m_OwnersPhone = i_OwnersPhone;
            r_Model = i_Model;
            r_RegistrationPlate = i_RegistrationPlate;
            m_EnergyType = i_EnergyType;
            m_EnergyLevel = i_EnergyLevel;
            m_CurrentAirPressure = i_CurrentAirPressure;
            m_NumberOfWheels = i_NumberOfWheels;
            m_VehicleType = i_VehicleType;
            m_Wheels = new List<Wheel>();
            setVehicleWheels(i_ManufacturerName, m_CurrentAirPressure, m_NumberOfWheels, i_MaxAirPressure);
        }

        private void setVehicleWheels(string i_ManufacturerName, float i_CurrentAirPressure, int i_NumberOfWheels, float i_MaxAirPressure)
        {
            for (int i = 0; i < i_NumberOfWheels; i++)
            {
                m_Wheels.Add(new Wheel(i_ManufacturerName, i_CurrentAirPressure, i_MaxAirPressure));
            }
        }

        public override string ToString()
        {
            string vehicleInfo = string.Format(
                                  "Owner's name: {1}{0}Owner's phone: {2}{0}Model: {3}{0}Registration Plate: {4}{0}Number Of Wheels: {5}{0}",
                                  Environment.NewLine,
                                  m_OwnersName,
                                  m_OwnersPhone,
                                  r_Model,
                                  r_RegistrationPlate,
                                  m_NumberOfWheels);
            return vehicleInfo + m_Wheels[0].ToString() + EnergyType.ToString();
        }

        public string OwnersName
        {
            get { return m_OwnersName; }
        }

        public string OwnersPhone
        {
            get { return m_OwnersPhone; }
        }

        public string Model
        {
            get { return r_Model; }
        }

        public string RegistrationPlate
        {
            get { return r_RegistrationPlate; }
        }

        public Energy EnergyType
        {
            get { return m_EnergyType; }
        }

        public float Energylevel
        {
            get { return m_EnergyLevel; }
        }

        public GarageENums.eVehicleType VehicleType
        {
            get { return m_VehicleType; }
        }

        public int NumberOfWheels
        {
            get { return m_NumberOfWheels; }
        }

        public List<Wheel> Wheels
        {
            get { return m_Wheels; }
        }

        public static List<VehicleProperties> GetRequiredProperties()
        {
            List<VehicleProperties> properties = new List<VehicleProperties>();
            properties.Add(new VehicleProperties(typeof(string), "Owner's Name"));
            properties.Add(new VehicleProperties(typeof(string), "Owner's Phone"));
            properties.Add(new VehicleProperties(typeof(string), "Vehicle Model"));
            properties.Add(new VehicleProperties(typeof(string), "Registration Plate"));
            properties.Add(new VehicleProperties(typeof(float), "Energy Level"));
            properties.Add(new VehicleProperties(typeof(float), "Current Air Pressure"));
            properties.Add(new VehicleProperties(typeof(string), "Wheels Manufacturer"));

            return properties;
        }
    }
}