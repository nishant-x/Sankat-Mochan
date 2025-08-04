import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Link } from 'react-router-dom';
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import './AddInventory.css';

const AddItem = () => {
    const [name, setName] = useState('');       
    const [quantity, setQuantity] = useState('');
    const [cost, setCost] = useState('');
    const [manufacturer, setManufacturer] = useState('');
    const [manufacturingDate, setManufacturingDate] = useState('');
    const [expiryDate, setExpiryDate] = useState('');
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const addItem = async (e) => {
        e.preventDefault();
        setLoading(true);
      
        // Validate fields
        if (!name || !quantity || !cost) {
            toast.error("Please provide valid item details!");
            setLoading(false);
            return;
        }
      
        if (isNaN(quantity) || isNaN(cost) || quantity <= 0 || cost <= 0) {
            toast.error("Quantity and cost must be valid positive numbers.");
            setLoading(false);
            return;
        }
      
        if (manufacturingDate && expiryDate && new Date(manufacturingDate) >= new Date(expiryDate)) {
            toast.error("Manufacturing date should be before expiry date.");
            setLoading(false);
            return;
        }

        try {
            // Check if the item already exists
            const checkResponse = await fetch(`http://localhost:5000/inventory/api/inventory/check?name=${name}`);
            const checkData = await checkResponse.json();

            if (checkResponse.ok && checkData.exists) {
                toast.warn(`This item is already present in the inventory (${checkData.count} times).`);
                setLoading(false);
                return;
            }

            // Proceed with adding the item
            const response = await fetch('http://localhost:5000/inventory/api/inventory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name,
                    quantity: parseInt(quantity),
                    manufacturer,
                    expiry_date: expiryDate,
                    manufacturing_date: manufacturingDate,
                    cost: parseFloat(cost),
                }),
            });

            const responseData = await response.json();

            if (!response.ok) {
                throw new Error(responseData.message || "Failed to add new item");
            }

            toast.success("Item added successfully!");
            navigate('/inventory');
        } catch (error) {
            toast.error(error.message || "Error adding item.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="AddItem-container">
            <h1 className="AddItem-heading">Add New Item</h1>
            <form onSubmit={addItem} className="AddItem-form">
                <label className="AddItem-label">Item Name:</label>
                <input type="text" value={name} onChange={(e) => setName(e.target.value)} required className="AddItem-input" />

                <label className="AddItem-label">Quantity:</label>
                <input type="number" value={quantity} onChange={(e) => setQuantity(e.target.value)} required min="1" className="AddItem-input" />

                <label className="AddItem-label">Cost:</label>
                <input type="number" value={cost} onChange={(e) => setCost(e.target.value)} required min="1" className="AddItem-input" />

                <label className="AddItem-label">Manufacturer:</label>
                <input type="text" value={manufacturer} onChange={(e) => setManufacturer(e.target.value)} className="AddItem-input" />

                <label className="AddItem-label">Manufacturing Date:</label>
                <input type="date" value={manufacturingDate} onChange={(e) => setManufacturingDate(e.target.value)} required className="AddItem-input" />

                <label className="AddItem-label">Expiry Date:</label>
                <input type="date" value={expiryDate} onChange={(e) => setExpiryDate(e.target.value)} required className="AddItem-input" />

                <div className="AddItem-button-container">
                    <button type="submit" className="AddItem-btn" disabled={loading}>
                        {loading ? 'Adding...' : 'Add Item'}
                    </button>
                    <Link to="/inventory" className="AddItem-btn">
                        Go back
                    </Link>
                </div>
            </form>
        </div>
    );
};

export default AddItem;
