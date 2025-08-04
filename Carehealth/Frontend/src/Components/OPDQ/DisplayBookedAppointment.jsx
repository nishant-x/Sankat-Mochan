import { useEffect, useState } from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./DisplayBookedAppointment.css";

function AppointmentsList() {
  const [appointments, setAppointments] = useState([]);

  useEffect(() => {
    fetch("http://localhost:5000/appointments")
      .then((response) => response.json())
      .then((data) => {
        setAppointments(data);
      })
      .catch((error) => {
        console.error("Error fetching appointments:", error);
        toast.error("Error fetching appointments");
      });
  }, []);

  return (
    <div className="display-appointment-container">
      <ToastContainer />
      <h2 className="display-appointment-heading">Appointments</h2>
      <table className="display-appointment-table" border="1">
        <thead>
          <tr>
            <th className="display-appointment-th">Name</th>
            <th className="display-appointment-th">Email</th>
            <th className="display-appointment-th">Phone</th>
            <th className="display-appointment-th">Date</th>
            <th className="display-appointment-th">Time</th>
            <th className="display-appointment-th">Type</th>
            <th className="display-appointment-th">Department</th>
          </tr>
        </thead>
        <tbody>
          {appointments.map((appointment, index) => (
            <tr key={index} className="display-appointment-row">
              <td className="display-appointment-td">{appointment.name}</td>
              <td className="display-appointment-td">{appointment.email}</td>
              <td className="display-appointment-td">{appointment.phone}</td>
              <td className="display-appointment-td">{appointment.date}</td>
              <td className="display-appointment-td">{appointment.time}</td>
              <td className="display-appointment-td">{appointment.type}</td>
              <td className="display-appointment-td">{appointment.department}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default AppointmentsList;
