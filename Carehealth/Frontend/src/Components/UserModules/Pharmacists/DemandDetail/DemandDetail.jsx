import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './DemandDetail.css';

const DemandDetails = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [demand, setDemand] = useState(null);
    const [prescriptions, setPrescriptions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [inventory, setInventory] = useState([]);
    const [inventoryLoading, setInventoryLoading] = useState(true);
    const [updatedQuantities, setUpdatedQuantities] = useState({});

    useEffect(() => {
        const fetchDemandDetails = async () => {
            try {
                const response = await fetch(`http://localhost:5000/inventory/api/demands/${id}`);
                if (!response.ok) throw new Error('Failed to fetch demand details');
                const data = await response.json();
                setDemand(data);
                setPrescriptions(data);
            } catch (error) {
                console.error("Error:", error);
                setError("Failed to fetch demand details.");
            } finally {
                setLoading(false);
            }
        };
        fetchDemandDetails();
    }, [id]);

    useEffect(() => {
        const fetchInventory = async () => {
            try {
                const response = await fetch('http://localhost:5000/inventory/api/inventory');
                if (!response.ok) throw new Error('Failed to fetch inventory.');
                const data = await response.json();
                setInventory(data);
            } catch (err) {
                console.error(err);
                setError('Failed to load inventory data.');
            } finally {
                setInventoryLoading(false);
            }
        };
        fetchInventory();
    }, []);

    const filteredInventory = prescriptions.map(prescription => {
        const match = inventory.find(item =>
            item.name.replace(/\s+/g, '').toLowerCase() === prescription.medicine_name.replace(/\s+/g, '').toLowerCase()
        );
        return match;
    }).filter(item => item);

    const handleInputChange = (inventoryId, value) => {
        const quantityToDeduct = parseInt(value);
        if (isNaN(quantityToDeduct)) return;
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
                const result = await response.json();
                if (!response.ok) {
                    throw new Error(result.message || 'Failed to update inventory');
                }
            });

            await Promise.all(inventoryUpdatePromises);

            const demandResponse = await fetch(`http://localhost:5000/inventory/api/demands/${id.trim()}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ handled_by_pharmacist: true, grand_total: grandTotal }),
            });

            if (!demandResponse.ok) {
                throw new Error('Failed to update demand status');
            }

            toast.success('Inventory and demand status updated successfully!');
            setDemand((prev) => ({
                ...prev,
                handled_by_pharmacist: true,
                grand_total: grandTotal
            }));
            navigate('/Hospital-demands');
        } catch (err) {
            console.error('Error in handleSubmitUpdate:', err);
            toast.error('Failed to update data. Please try again.');
            setError('Failed to update data. Please try again.');
        }
    };

    if (loading || inventoryLoading) return <div>Loading details...</div>;
    if (error) return <div>{error}</div>;
    if (!demand) return <div>Demand not found!</div>;

    return (
        <div className="demand-details-container">
            <ToastContainer />
            <header className="demand-details-header">Demand & Inventory Management</header>
            <div className="demand-details-content">
                <div className="demand-details-left-section">
                    <h3 className="demand-details-title">{demand.demand_id}'s Demand</h3>
                    <table className="demand-details-table">
                        <thead>
                            <tr>
                                <th>Medicine Name</th>
                                <th>Quantity</th>
                            </tr>
                        </thead>
                        <tbody>
                            {prescriptions.map((item, index) => (
                                <tr key={index}>
                                    <td>{item.medicine_name}</td>
                                    <td>{item.quantity}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <button className="demand-details-back-btn" onClick={() => navigate(-1)}>Go Back</button>
                </div>
                <div className="demand-details-right-section">
                    <h3 className="demand-details-title">Inventory Management</h3>
                    <table className="demand-details-inventory-table">
                        <thead>
                            <tr>
                                <th>Item Name</th>
                                <th>Item Cost</th>
                                <th>Current Quantity</th>
                                <th>Update Quantity</th>
                                <th>Cost</th>
                            </tr>
                        </thead>
                        <tbody>
                            {prescriptions.map((prescription, index) => {
                                const item = filteredInventory[index];
                                return (
                                    <tr key={index}>
                                        <td>{prescription.medicine_name}</td>
                                        <td>₹{item ? item.cost : "N/A"}</td>
                                        <td>{item ? (item.quantity > 0 ? item.quantity : "Out of Stock") : "Unavailable"}</td>
                                        <td>{item && item.quantity > 0 ? (
                                            <input
                                                type="number"
                                                min="0"
                                                value={updatedQuantities[item._id] || ''}
                                                onChange={(e) => handleInputChange(item._id, e.target.value)}
                                                className="demand-details-input"
                                            />
                                        ) : "N/A"}</td>
                                        <td>{item && item.quantity > 0 ? `₹${(updatedQuantities[item._id] || 0) * item.cost}` : "₹0"}</td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                    <h3 className="demand-details-grand-total">Grand Total: ₹{calculateTotalCost()}</h3>
                    <button onClick={handleSubmitUpdate} className="demand-details-submit-btn">Submit</button>
                </div>
            </div>
        </div>
    );
};

export default DemandDetails;