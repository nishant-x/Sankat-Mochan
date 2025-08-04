import React, { useState, useEffect } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./AppointmentsList.css";

const predefinedSlots = ["09:00 AM", "09:30 AM", "10:00 AM", "10:30 AM", "11:00 AM"];

const AppointmentsList = () => {
    const [appointments, setAppointments] = useState([]);
    const [loading, setLoading] = useState(false);
    const [rescheduleData, setRescheduleData] = useState({ id: "", date: "", time: "" });

    useEffect(() => {
        fetchAppointments();
    }, []);

    const fetchAppointments = async () => {
        try {
            const response = await axios.get("http://localhost:5000/opdRoutes/appointments");
            setAppointments(response.data);
        } catch (error) {
            console.error("Error fetching appointments:", error);
            toast.error("Error fetching appointments");
        }
    };

    // Cancel appointment
    const cancelAppointment = async (id) => {
        setLoading(true);
        try {
            await axios.delete(`http://localhost:5000/opdRoutes/cancel-appointment/${id}`);
            toast.success("Appointment cancelled successfully");
            fetchAppointments();
        } catch (error) {
            toast.error("Error cancelling appointment");
        }
        setLoading(false);
    };

    // Reschedule appointment with availability check
    const rescheduleAppointment = async (e) => {
        e.preventDefault();

        if (!rescheduleData.date || !rescheduleData.time) {
            toast.error("Please select a new date and time");
            return;
        }

        try {
            // Check if the slot is available before rescheduling
            const response = await axios.post("http://localhost:5000/opdRoutes/check-availability", {
                date: rescheduleData.date,
                time: rescheduleData.time,
            });

            if (!response.data.available) {
                toast.error("Selected time slot is already taken! Choose another.");
                return;
            }

            setLoading(true);
            await axios.put(`http://localhost:5000/opdRoutes/reschedule-appointment/${rescheduleData.id}`, {
                date: rescheduleData.date,
                time: rescheduleData.time,
            });

            toast.success("Appointment rescheduled successfully");
            fetchAppointments();
        } catch (error) {
            toast.error("Error rescheduling appointment");
        } finally {
            setLoading(false);
            setRescheduleData({ id: "", date: "", time: "" });
        }
    };

    return (
        <div>
            <ToastContainer />
            <h2>Appointments</h2>
            {appointments.length === 0 ? (
                <p>No appointments found.</p>
            ) : (
                <ul className="appointments-list">
                    {appointments.map((appointment) => (
                        <li key={appointment._id}>
                            <strong>{appointment.name}</strong> - {appointment.date} at {appointment.time}
                            <div className="btn-appointment">
                                <button onClick={() => cancelAppointment(appointment._id)} disabled={loading}>
                                    Cancel
                                </button>
                                <button onClick={() => setRescheduleData({ id: appointment._id, date: "", time: "" })}>
                                    Reschedule
                                </button>
                            </div>
                        </li>
                    ))}
                </ul>
            )}

{rescheduleData.id && (
    <div className="reschedule-container">
        <h3 className="reschedule-title">Reschedule Appointment</h3>
        
        <label className="reschedule-label">New Date:</label>
        <input
            type="date"
            className="reschedule-input"
            onChange={(e) => setRescheduleData({ ...rescheduleData, date: e.target.value })}
            required
        />

        <label className="reschedule-label">New Time Slot:</label>
        <select
            className="reschedule-select"
            onChange={(e) => setRescheduleData({ ...rescheduleData, time: e.target.value })}
            required
        >
            <option value="">Select a time slot</option>
            {predefinedSlots.map((slot, index) => (
                <option key={index} value={slot}>
                    {slot}
                </option>
            ))}
        </select>

        <div className="reschedule-btn-group">
            <button className="btn btn-confirm" onClick={rescheduleAppointment} disabled={loading}>
                Confirm Reschedule
            </button>
            <button className="btn btn-cancel-reschedule" onClick={() => setRescheduleData({ id: "", date: "", time: "" })}>
                Cancel
            </button>
        </div>
    </div>
)}

        </div>
    );
};

export default AppointmentsList;
