import { useEffect, useState } from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./Appointment-count.css";

const TodayAppointments = () => {
    const [appointments, setAppointments] = useState([]);

    useEffect(() => {
        const fetchAppointments = async () => {
            try {
                const response = await fetch("http://localhost:5000/api/appointments/today");
                const data = await response.json();
                setAppointments(data);
            } catch (error) {
                console.error("Error fetching appointments:", error);
                toast.error("Error fetching appointments");
            }
        };
        fetchAppointments();
    }, []);

    return (
        <div className="appointment-container">
            <ToastContainer />
            <h2 className="appointment-heading">Today's Appointments</h2>
            <table className="appointment-table">
                <thead>
                    <tr className="appointment-header-row">
                        <th className="appointment-header">Name</th>
                        <th className="appointment-header">Doctor</th>
                        <th className="appointment-header">Department</th>
                        <th className="appointment-header">Type</th>
                        <th className="appointment-header">Time</th>
                        <th className="appointment-header">Reached</th>
                    </tr>
                </thead>
                <tbody>
                    {appointments.length > 0 ? (
                        appointments.map((appointment) => (
                            <tr key={appointment._id} className="appointment-row">
                                <td className="appointment-cell">{appointment.name}</td>
                                <td className="appointment-cell">{appointment.doctorName}</td>
                                <td className="appointment-cell">{appointment.department}</td>
                                <td className="appointment-cell">{appointment.type}</td>
                                <td className="appointment-cell">{appointment.time}</td>
                                <td className="appointment-cell">{appointment.reached ? "Yes" : "No"}</td>
                            </tr>
                        ))
                    ) : (
                        <tr className="appointment-empty-row">
                            <td colSpan="6" className="appointment-empty-cell">
                                No appointments today
                            </td>
                        </tr>
                    )}
                </tbody>
            </table>
        </div>
    );
};

export default TodayAppointments;
