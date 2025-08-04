import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './InventoryManagement.css';

const InventoryManagement = () => {
    const [items, setItems] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchItems();
    }, []);

    const fetchItems = async () => {
        setLoading(true);
        try {
            const response = await fetch('http://localhost:5000/inventory/api/inventory');
            if (!response.ok) {
                const message = await response.json();
                throw new Error(`Failed to fetch items: ${message.error}`);
            }
            const data = await response.json();
            setItems(data);
            toast.success("Inventory loaded successfully!"); // Success toast
            checkForShortages(data);
        } catch (error) {
            setError(`Failed to fetch items: ${error.message}`);
            console.error('Error fetching items:', error);
            toast.error("Error fetching inventory."); // Error toast
        } finally {
            setLoading(false);
        }
    };

    const checkForShortages = (items) => {
        const shortageThreshold = 10; 
        items.forEach((item) => {
            if (item.quantity < shortageThreshold) {
                sendMedicineShortageEmail("sighaikrish769@gmail.com", item.name, item.quantity);
                toast.warn(`Low stock alert: ${item.name} has only ${item.quantity} left!`);
            }
        });
    };

    const sendMedicineShortageEmail = async (pharmacistEmail, medicineName, currentStock) => {
        try {
            const response = await fetch("http://localhost:5000/send-email", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ pharmacistEmail, medicineName, currentStock }),
            });

            if (!response.ok) {
                throw new Error("Failed to send email");
            }

            const data = await response.json();
            console.log("Email sent: ", data.message);
            toast.info(`Shortage email sent for ${medicineName}`);
        } catch (error) {
            console.error("Error sending email: ", error);
            toast.error("Failed to send shortage email.");
        }
    };

    const formatDate = (dateString) => {
        return dateString ? dateString.split('T')[0] : '';
    };

    return (
        <div className="inventorymanagement-container">
            <ToastContainer />
            <h1>Manage Inventory</h1>

            <div className="inventorymanagement-button-container">
                <Link to="/add-inventory" className="inventorymanagement-add-btn">Add Item</Link>
                <Link to="/expired-product" className="inventorymanagement-expired-products-btn">
                    View Expired Products
                </Link>
                <Link to="/about-to-expire" className="inventorymanagement-abouttoexpire-btn">
                    View About to Expire Products
                </Link>
                <Link to="/pharmacy-dashboard" className="inventorymanagement-home-btn">
                    Home
                </Link>
            </div>
            <br />

            {loading ? (
                <p>Loading...</p>
            ) : error ? (
                <p className="inventorymanagement-error-message">{error}</p>
            ) : items.length > 0 ? (
                <table className="inventorymanagement-item-list">
                    <thead>
                        <tr>
                            <th>Item Name</th>
                            <th>Quantity</th>
                            <th>Cost</th>
                            <th>Manufacture Date</th>
                            <th>Expiry Date</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {items.map((item) => (
                            <tr key={item._id}>
                                <td>{item.name}</td>
                                <td>{item.quantity}</td>
                                <td>{item.cost}</td>
                                <td>{formatDate(item.manufacturing_date)}</td>
                                <td>{formatDate(item.expiry_date)}</td>
                                <td>
                                    <Link to={`/update-quantity/${item._id}`} className="inventorymanagement-update-quantity-btn">
                                        Update Quantity
                                    </Link>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            ) : (
                <p>No items available in inventory.</p>
            )}
        </div>
    );
};

export default InventoryManagement;
