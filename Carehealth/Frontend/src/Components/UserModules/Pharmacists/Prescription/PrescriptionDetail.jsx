import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './PrescriptionDetail.css';



const PrescriptionDetails = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [patient, setPatient] = useState(null);
    const [prescriptions, setPrescriptions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [inventory, setInventory] = useState([]);
    const [inventoryLoading, setInventoryLoading] = useState(true);
    const [updatedQuantities, setUpdatedQuantities] = useState({});

    useEffect(() => {
        const fetchPatientDetails = async () => {
            try {
                const response = await fetch(`http://localhost:5000/inventory/api/patients/${id.trim()}`);
                if (!response.ok) throw new Error('Failed to fetch patient details.');
                const data = await response.json();
                setPatient(data);
                setPrescriptions(formatPrescriptions(data.prescription));
            } catch (err) {
                setError('Failed to load patient details.');
                toast.error('Failed to load patient details.');
            } finally {
                setLoading(false);
            }
        };
        fetchPatientDetails();
    }, [id]);

    useEffect(() => {
        const fetchInventory = async () => {
            try {
                const response = await fetch('http://localhost:5000/inventory/api/inventory');
                if (!response.ok) throw new Error('Failed to fetch inventory.');
                const data = await response.json();
                setInventory(data);
            } catch (err) {
                setError('Failed to load inventory data.');
                toast.error('Failed to load inventory data.');
            } finally {
                setInventoryLoading(false);
            }
        };
        fetchInventory();
    }, []);

    const formatPrescriptions = (prescription) => {
        if (Array.isArray(prescription)) {
            return prescription;
        } else if (typeof prescription === 'string') {
            return prescription.split(',').map((item) => {
                const [name, quantity] = item.split(':');
                return { medicine_name: name.trim(), quantity: parseInt(quantity) };
            });
        } else if (typeof prescription === 'object') {
            return Object.entries(prescription).map(([name, quantity]) => ({
                medicine_name: name.trim(),
                quantity,
            }));
        }
        return [];
    };

    const handleInputChange = (inventoryId, value) => {
        const quantityToDeduct = parseInt(value);
        if (isNaN(quantityToDeduct) || quantityToDeduct < 0) return;
        setUpdatedQuantities((prev) => ({
            ...prev,
            [inventoryId]: quantityToDeduct,
        }));
    };

    const calculateTotalCost = () => {
        return Object.entries(updatedQuantities).reduce((total, [inventoryId, quantity]) => {
            const item = inventory.find(item => item._id === inventoryId);
            return total + (item ? item.cost * quantity : 0);
        }, 0);
    };

    const handleSubmitUpdate = async () => {
        const grandTotal = calculateTotalCost();

        try {
            const inventoryUpdatePromises = Object.entries(updatedQuantities).map(async ([inventoryId, quantityToDeduct]) => {
                const inventoryItem = inventory.find(item => item._id === inventoryId);
                if (!inventoryItem) return;
                const newQuantity = Math.max(inventoryItem.quantity - quantityToDeduct, 0);
                const response = await fetch(`http://localhost:5000/inventory/api/inventory/${inventoryId}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ quantity: newQuantity }),
                });
                if (!response.ok) {
                    throw new Error('Failed to update inventory');
                }
            });

            await Promise.all(inventoryUpdatePromises);

            const patientResponse = await fetch(`http://localhost:5000/inventory/api/patients/${id.trim()}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ handled_by_pharmacist: true, grand_total: grandTotal }),
            });

            if (!patientResponse.ok) {
                throw new Error('Failed to update patient status');
            }

            setPatient((prev) => ({
                ...prev,
                handled_by_pharmacist: true,
                grand_total: grandTotal
            }));

            toast.success('Inventory updated successfully!');
            navigate('/patient-list');
        } catch (err) {
            setError('Failed to update data. Please try again.');
            toast.error('Failed to update inventory.');
        }
    };

    if (loading || inventoryLoading) return <div>Loading details...</div>;
    if (error) return <div>{error}</div>;
    if (!patient) return <div>Patient not found!</div>;
    
    return (
        <div className="Prescription-container">
            <header>Prescription & Inventory Management</header>
            <div className="Prescription-content">
                <h3>{patient.patient_name}'s Prescription</h3>
                <h3>Grand Total: ₹{calculateTotalCost()}</h3>
                <button onClick={handleSubmitUpdate} className="Prescription-submit-btn">Submit</button>
            </div>
        </div>
    );
};

export default PrescriptionDetails;