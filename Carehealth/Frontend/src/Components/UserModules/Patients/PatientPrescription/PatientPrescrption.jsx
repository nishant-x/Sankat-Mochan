import React, { useState, useEffect } from 'react';
import { useLocation } from "react-router-dom";
import "./PatientPrescrption.css"

const PatientPrescrption = () => {
  const location = useLocation();
  const { userDetails } = location.state || {}; // Get userDetails from location.state
  const [prescriptions, setPrescriptions] = useState(null);

  console.log("from prescriptions", userDetails);

  useEffect(() => {
    const fetchPrescriptions = async () => {
      if (!userDetails?._id && !userDetails) {
        console.error('User ID is missing');
        return;
      }
      try {
        const response = await fetch(`http://localhost:5000/prescriptions/patient/prescription/${ userDetails.patientId || userDetails._id || userDetails}`);
        const data = await response.json();
    
        console.log('Full fetched response:', data); // Log the full response
    
        if (response.ok) {
          setPrescriptions(data); // Assuming `data` is the array of prescriptions
        } else {
          console.error('Error from server:', data.message || 'Failed to fetch prescriptions');
        }
      } catch (err) {
        console.error('Error fetching prescriptions:', err);
      }
    };
    
    fetchPrescriptions();
  }, [userDetails?._id]); // Re-run the effect if userDetails._id changes

  if (!userDetails) {
    return <div>No user details found.</div>;
  }

  return (
    <div className="patient-prescription-container">
      <h1 className="patient-prescription-title">
        {userDetails?.name ? `Prescriptions for ${userDetails.name}` : "Prescriptions Of Patient"}
      </h1>

      {prescriptions ? (
        prescriptions.length > 0 ? (
          prescriptions.map((prescription, index) => (
            <div key={prescription._id} className="patient-prescription-card">
              <h2 className="patient-prescription-subtitle">Prescription {index + 1}</h2>
              <p><strong>Date:</strong> {new Date(prescription.prescription_date).toLocaleDateString()}</p>

              <h3 className="patient-prescription-medicine-title">Medicines:</h3>
              {prescription.prescription.length > 0 ? (
                <ul className="patient-prescription-medicine-list">
                  {prescription.prescription.map((med, medIndex) => (
                    <li key={medIndex} className="patient-prescription-medicine-item">
                      <strong>Medicine:</strong> {med.medicine_name} <br />
                      <strong>Quantity:</strong> {med.quantity} <br />
                      <strong>Dosage:</strong> {med.dosage} <br />
                      <strong>Duration:</strong> {med.duration}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="patient-prescription-no-medicine">No medicines found in this prescription.</p>
              )}
            </div>
          ))
        ) : (
          <p className="patient-prescription-no-data">No prescriptions found for this patient.</p>
        )
      ) : (
        <p className="patient-prescription-loading">Loading prescriptions...</p>
      )}
    </div>
  );
};

export default PatientPrescrption;
