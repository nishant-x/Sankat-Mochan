import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './UpdateQuantity.css';

const UpdateQuantity = () => {
    const { id } = useParams();
    const [quantity, setQuantity] = useState('');
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        fetchMedicine();
    }, []);

    const fetchMedicine = async () => {
        setLoading(true);
        try {
            const response = await fetch(`http://localhost:5000/inventory/api/inventory/${id}`);
            if (!response.ok) {
                throw new Error('Failed to fetch medicine');
            }
            const data = await response.json();
            setQuantity(data.quantity);
        } catch (error) {
            toast.error('Failed to fetch medicine.', { position: 'top-right' });
            console.error('Error fetching medicine:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleUpdate = async (e) => {
        e.preventDefault();
        if (isNaN(quantity) || quantity < 0) {
            toast.error('Quantity must be a valid non-negative number.', { position: 'top-right' });
            return;
        }
        setLoading(true);
        try {
            const response = await fetch(`http://localhost:5000/inventory/api/inventory/${id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ quantity: quantity })
            });
            if (!response.ok) {
                throw new Error('Failed to update quantity');
            }
            toast.success('Quantity updated successfully!', { position: 'top-right' });
            navigate('/inventory');
        } catch (error) {
            toast.error('Failed to update quantity.', { position: 'top-right' });
            console.error('Error updating quantity:', error);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div className="Update-quantity-container">
            <h1 className="Update-quantity-heading">Update Quantity</h1>
            {loading ? (
                <p className="Update-quantity-loading">Loading...</p>
            ) : (
                <form onSubmit={handleUpdate} className="Update-quantity-form">
                    <p className="Update-quantity-current-quantity">Current Quantity: {quantity}</p>
                    <label htmlFor="quantity" className="Update-quantity-label">Quantity</label>
                    <input
                        type="number"
                        id="quantity"
                        value={quantity}
                        onChange={(e) => setQuantity(Number(e.target.value))}
                        required
                        className="Update-quantity-input-field"
                    />
                    <button type="submit" className="Update-quantity-update-btn">Update</button>
                </form>
            )}
        </div>
    );
};

export default UpdateQuantity;
