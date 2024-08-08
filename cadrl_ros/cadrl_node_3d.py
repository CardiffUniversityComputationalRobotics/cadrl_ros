import rclpy
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, Bool
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from ford_msgs.msg import PlannerMode, Clusters
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from rclpy.duration import Duration

from launch_ros.substitutions import FindPackageShare

import numpy as np
import copy
import time

from cadrl_ros import network, agent, util


PED_RADIUS = 0.35


# angle_1 - angle_2
# contains direction in range [-3.14, 3.14]
def find_angle_diff(angle_1, angle_2):
    angle_diff_raw = angle_1 - angle_2
    angle_diff = (angle_diff_raw + np.pi) % (2 * np.pi) - np.pi
    return angle_diff


class NN_tb3(Node):
    def __init__(self, veh_name, veh_data, nn, actions):
        super().__init__("nn_tb3")

        # tb3

        # canon
        self.prev_other_agents_state = []

        # vehicle info
        self.veh_name = veh_name
        self.veh_data = veh_data

        # neural network
        self.nn = nn
        self.actions = actions
        # self.value_net = value_net
        self.operation_mode = PlannerMode()
        self.operation_mode.mode = self.operation_mode.NN

        # for subscribers
        self.pose = PoseStamped()
        self.vel = Vector3()
        self.psi = 0.0
        self.ped_traj_vec = []
        self.other_agents_state = []

        # for publishers
        self.global_goal = PoseStamped()
        self.goal = PoseStamped()
        self.goal.pose.position.x = veh_data["goal"][0]
        self.goal.pose.position.y = veh_data["goal"][1]
        self.desired_position = PoseStamped()
        self.desired_action = np.zeros((2,))

        # handle obstacles close to vehicle's front
        self.stop_moving_flag = True
        self.d_min = 0.3
        self.new_global_goal_received = True

        # visualization
        self.path_marker = Marker()

        # Clusters
        self.prev_clusters = Clusters()
        self.current_clusters = Clusters()

        # subscribers and publishers
        self.num_poses = 0
        self.num_actions_computed = 0.0

        #! PUBLISHERS TOPICS

        self.declare_parameter("cmd_vel_topic", "/pepper/cmd_vel")
        self.cmd_vel_topic = (
            self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        )

        # self.pub_others = self.create_publisher(Vector3, '~other_vels', 1)
        self.pub_twist = self.create_publisher(Twist, self.cmd_vel_topic, 1)
        self.pub_pose_marker = self.create_publisher(Marker, "/pose_marker", 1)
        self.pub_agent_marker = self.create_publisher(Marker, "/agent_marker", 1)
        self.pub_agent_markers = self.create_publisher(MarkerArray, "/agent_markers", 1)
        self.pub_path_marker = self.create_publisher(Marker, "/path_marker", 1)
        self.pub_goal_path_marker = self.create_publisher(
            Marker, "/goal_path_marker", 1
        )
        #! SUBSCRIBERS TOPICS

        self.declare_parameter("odom_topic", "/pepper/odom_groundtruth")
        self.odom_topic = (
            self.get_parameter("odom_topic").get_parameter_value().string_value
        )

        self.sub_pose = self.create_subscription(
            Odometry, self.odom_topic, self.cbPose, 1
        )
        self.sub_mode = self.create_subscription(
            PlannerMode, "/planner_fsm/mode", self.cbPlannerMode, 1
        )
        # self.sub_global_goal = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.cbGlobalGoal, 1)
        self.sub_global_goal = self.create_subscription(
            PoseStamped, "/subgoal", self.cbGlobalGoal, 1
        )
        self.sub_subgoal = self.create_subscription(
            PoseStamped, "/subgoal", self.cbSubGoal, 1
        )
        self.result_sub = self.create_subscription(
            Bool, "/cadrl_result", self.cbResult, 1
        )

        # subgoals
        self.sub_goal = Vector3()

        self.use_clusters = True
        # self.use_clusters = False
        if self.use_clusters:
            self.sub_clusters = self.create_subscription(
                Clusters, "/clusters", self.cbClusters, 1
            )
        else:
            print("no peds")

        # control timer
        self.control_timer = self.create_timer(0.01, self.cbControl)
        self.nn_timer = self.create_timer(0.1, self.cbComputeActionGA3C)

    def cbResult(self, msg):
        print("got result stopping robot")
        self.stop_moving_flag = True

    def cbGlobalGoal(self, msg):
        # self.stop_moving_flag = True
        # self.new_global_goal_received = True
        self.global_goal = msg
        self.operation_mode.mode = self.operation_mode.SPIN_IN_PLACE
        self.goal.pose.position.x = msg.pose.position.x
        self.goal.pose.position.y = msg.pose.position.y
        self.goal.header = msg.header

        # reset subgoals
        # print("new goal: "+str([self.goal.pose.position.x,self.goal.pose.position.y]))

    def cbSubGoal(self, msg):
        print("goal subgoal")

        self.stop_moving_flag = False
        self.sub_goal.x = msg.pose.position.x
        self.sub_goal.y = msg.pose.position.y
        # print("new subgoal: "+str(self.sub_goal))

    def cbPlannerMode(self, msg):
        self.operation_mode = msg
        self.operation_mode.mode = self.operation_mode.NN

    def cbPose(self, msg):
        self.cbVel(msg)
        self.num_poses += 1
        q = msg.pose.pose.orientation
        self.psi = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
        )  # bounded by [-pi, pi]
        self.pose = msg.pose
        self.visualize_pose(msg.pose.pose.position, msg.pose.pose.orientation)

    def cbVel(self, msg):
        self.vel = msg.twist.twist.linear

    def cbClusters(self, msg):
        # print(msg)
        other_agents = []

        xs = []
        ys = []
        radii = []
        labels = []
        num_clusters = len(msg.mean_points)
        # print(num_clusters)
        for i in range(num_clusters):
            index = msg.labels[i]
            x = msg.mean_points[i].x
            y = msg.mean_points[i].y
            v_x = msg.velocities[i].x
            v_y = msg.velocities[i].y
            inflation_factor = 1.5

            # radius = msg.mean_points[i].z*inflation_factor
            radius = 0.2

            xs.append(x)
            ys.append(y)
            radii.append(radius)
            labels.append(index)

            # self.visualize_other_agent(x,y,radius,msg.labels[i])
            # helper fields
            heading_angle = np.arctan2(v_y, v_x)
            pref_speed = np.linalg.norm(np.array([v_x, v_y]))
            goal_x = x + 5.0
            goal_y = y + 5.0

            # v_x = 3*v_x; v_y = 3*v_y

            if pref_speed < 0.2:
                pref_speed = 0
                v_x = 0
                v_y = 0
            other_agents.append(
                agent.Agent(
                    x, y, goal_x, goal_y, radius, pref_speed, heading_angle, index
                )
            )
        self.visualize_other_agents(xs, ys, radii, labels)
        self.other_agents_state = other_agents

    def stop_moving(self):
        twist = Twist()
        self.pub_twist.publish(twist)

    def update_action(self, action):
        # print 'update action'
        self.desired_action = action
        self.desired_position.pose.position.x = self.pose.pose.position.x + 1 * action[
            0
        ] * np.cos(action[1])
        self.desired_position.pose.position.y = self.pose.pose.position.y + 1 * action[
            0
        ] * np.sin(action[1])

    def find_vmax(self, d_min, heading_diff):
        # Calculate maximum linear velocity, as a function of error in
        # heading and clear space in front of the vehicle
        # (With nothing in front of vehicle, it's not important to
        # track MPs perfectly; with an obstacle right in front, the
        # vehicle must turn in place, then drive forward.)
        d_min = max(0.0, d_min)
        x = 0.3
        margin = 0.3
        # y = max(d_min - 0.3, 0.0)
        y = max(d_min, 0.0)
        # making sure x < y
        if x > y:
            x = 0
        w_max = 1
        # x^2 + y^2 = (v_max/w_max)^2
        v_max = w_max * np.sqrt(x**2 + y**2)
        v_max = np.clip(v_max, 0.0, self.veh_data["pref_speed"])
        # print 'V_max, x, y, d_min', v_max, x, y, d_min
        if abs(heading_diff) < np.pi / 18:
            return self.veh_data["pref_speed"]
        return v_max

    def cbControl(self):

        # print("callback control")

        if self.stop_moving_flag:
            # print(self.goal.header.stamp)
            self.stop_moving()
            return
        elif self.operation_mode.mode == self.operation_mode.NN:
            print("moving operation to goal")
            desired_yaw = self.desired_action[1]
            yaw_error = desired_yaw - self.psi
            if abs(yaw_error) > np.pi:
                yaw_error -= np.sign(yaw_error) * 2 * np.pi

            gain = 1.3  # canon: 2
            vw = gain * yaw_error

            use_d_min = True
            if use_d_min:  # canon: True
                # use_d_min = True
                # print "vmax:", self.find_vmax(self.d_min,yaw_error)
                vx = min(self.desired_action[0], self.find_vmax(self.d_min, yaw_error))
            else:
                vx = self.desired_action[0]

            twist = Twist()
            twist.angular.z = vw
            twist.linear.x = vx
            # print(twist)
            self.pub_twist.publish(twist)
            self.visualize_action(use_d_min)
            return

        elif self.operation_mode.mode == self.operation_mode.SPIN_IN_PLACE:
            # print('Spinning in place.')
            # self.stop_moving_flag = False
            angle_to_goal = np.arctan2(
                self.global_goal.pose.position.y - self.pose.pose.position.y,
                self.global_goal.pose.position.x - self.pose.pose.position.x,
            )
            global_yaw_error = self.psi - angle_to_goal
            if abs(global_yaw_error) > 0.5:
                print("spinning in place")
                vx = 0.0
                vw = 1.0
                twist = Twist()
                twist.angular.z = vw
                twist.linear.x = vx
                self.pub_twist.publish(twist)
                # print twist
            else:
                # print('Done spinning in place')
                self.operation_mode.mode = self.operation_mode.NN
                # self.new_global_goal_received = False
            return
        else:
            self.stop_moving()
            return

    def cbComputeActionGA3C(self):
        if self.operation_mode.mode != self.operation_mode.NN or self.stop_moving_flag:
            # print 'Not in NN mode'
            # print(self.stop_moving_flag)
            return

        # construct agent_state
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        v_x = self.vel.x
        v_y = self.vel.y
        radius = self.veh_data["radius"]
        heading_angle = self.psi
        pref_speed = self.veh_data["pref_speed"]
        goal_x = self.sub_goal.x
        goal_y = self.sub_goal.y
        marker_goal = [goal_x, goal_y]
        self.visualize_subgoal(marker_goal, None)

        # in case current speed is larger than desired speed
        # print goal_x+goal_y
        v = np.linalg.norm(np.array([v_x, v_y]))
        if v > pref_speed:
            v_x = v_x * pref_speed / v
            v_y = v_y * pref_speed / v

        host_agent = agent.Agent(
            x, y, goal_x, goal_y, radius, pref_speed, heading_angle, 0
        )
        host_agent.vel_global_frame = np.array([v_x, v_y])
        # host_agent.print_agent_info()

        other_agents_state = copy.deepcopy(self.other_agents_state)
        obs = host_agent.observe(other_agents_state)[1:]
        obs = np.expand_dims(obs, axis=0)

        # predictions = self.nn.predict_p(obs,None)[0]
        predictions = self.nn.predict_p(obs)[0]

        raw_action = copy.deepcopy(self.actions[np.argmax(predictions)])
        action = np.array(
            [pref_speed * raw_action[0], util.wrap(raw_action[1] + self.psi)]
        )

        # if close to goal
        kp_v = 0.5
        kp_r = 1

        goal_tol = 0.1

        if host_agent.dist_to_goal < 2.0:  # and self.percentComplete>=0.9:
            # print "somewhat close to goal"
            pref_speed = max(
                min(kp_v * (host_agent.dist_to_goal - 0.1), pref_speed), 0.0
            )
            action[0] = min(raw_action[0], pref_speed)
            turn_amount = (
                max(min(kp_r * (host_agent.dist_to_goal - 0.1), 1.0), 0.0)
                * raw_action[1]
            )
            action[1] = util.wrap(turn_amount + self.psi)
        if host_agent.dist_to_goal < goal_tol:
            # current goal, reached, increment for next goal
            print("===============\ngoal reached: " + str([goal_x, goal_y]))
            # self.stop_moving_flag = True
            self.new_global_goal_received = False
            self.stop_moving()
            # self.goal_idx += 1
        else:
            pass
            # self.stop_moving_flag = False

        # print(action)
        self.update_action(action)

    def update_subgoal(self, subgoal):
        self.goal.pose.position.x = subgoal[0]
        self.goal.pose.position.y = subgoal[1]

    def visualize_subgoal(self, subgoal, subgoal_options=None):
        markers = MarkerArray()

        # Display GREEN DOT at NN subgoal
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "subgoal"
        marker.id = 0
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.pose.position.x = subgoal[0]
        marker.pose.position.y = subgoal[1]
        marker.scale = Vector3(x=0.2, y=0.2, z=0)
        marker.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)
        marker.lifetime = Duration(seconds=2.0)
        self.pub_goal_path_marker.publish(marker)

        if subgoal_options is not None:
            for i in range(len(subgoal_options)):
                marker = Marker()
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.header.frame_id = "map"
                marker.ns = "subgoal"
                marker.id = i + 1
                # marker.type = marker.CUBE
                marker.type = marker.CYLINDER
                marker.action = marker.ADD
                marker.pose.position.x = subgoal_options[i][0]
                marker.pose.position.y = subgoal_options[i][1]
                marker.scale = Vector3(x=0.2, y=0.2, z=0.2)
                marker.color = ColorRGBA(r=0.0, g=0.0, b=255, a=1.0)
                marker.lifetime = Duration(seconds=1.0)
                self.pub_goal_path_marker.publish(marker)

    def visualize_pose(self, pos, orientation):
        # Yellow Box for Vehicle
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "agent"
        marker.id = 0
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.position = pos
        marker.pose.orientation = orientation
        marker.scale = Vector3(x=0.7, y=0.42, z=1)
        marker.color = ColorRGBA(r=1.0, g=1.0, a=1.0)
        marker.lifetime = Duration(seconds=1.0)
        self.pub_pose_marker.publish(marker)

        # Red track for trajectory over time
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "agent"
        marker.id = self.num_poses
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.position = pos
        marker.pose.orientation = orientation
        marker.scale = Vector3(x=0.2, y=0.2, z=0.2)
        marker.color = ColorRGBA(r=1.0, a=1.0)
        marker.lifetime = Duration(seconds=10.0)
        self.pub_pose_marker.publish(marker)

        # print marker

    def visualize_other_agents(self, xs, ys, radii, labels):
        markers = MarkerArray()
        for i in range(len(xs)):
            # Orange box for other agent
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "map"
            marker.ns = "other_agent"
            marker.id = labels[i]
            marker.type = marker.CYLINDER
            marker.action = marker.ADD
            marker.pose.position.x = xs[i]
            marker.pose.position.y = ys[i]
            # marker.pose.orientation = orientation
            marker.scale = Vector3(x=2 * radii[i], y=2 * radii[i], z=1)
            # if labels[i] <= 23: # for static map
            #     # print sm
            #     marker.color = ColorRGBA(r=0.5,g=0.4,a=1.0)
            # else:
            marker.color = ColorRGBA(r=1.0, g=0.4, a=1.0)
            marker.lifetime = Duration(seconds=0.5)
            markers.markers.append(marker)

        self.pub_agent_markers.publish(markers)

    def visualize_action(self, use_d_min):
        # Display BLUE ARROW from current position to NN desired position
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "path_arrow"
        marker.id = 0
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.points.append(self.pose.pose.position)
        marker.points.append(self.desired_position.pose.position)
        marker.scale = Vector3(x=0.1, y=0.2, z=0.2)
        marker.color = ColorRGBA(b=1.0, a=1.0)
        marker.lifetime = Duration(seconds=0.1)
        self.pub_goal_path_marker.publish(marker)

        # Display BLUE DOT at NN desired position
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "path_trail"
        marker.id = self.num_poses
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.position = copy.deepcopy(self.desired_position.pose.position)
        marker.scale = Vector3(x=0.2, y=0.2, z=0.4)
        marker.color = ColorRGBA(b=1.0, a=0.1)
        marker.lifetime = Duration(seconds=100)
        if self.desired_action[0] == 0.0:
            marker.pose.position.x += 2.0 * np.cos(self.desired_action[1])
            marker.pose.position.y += 2.0 * np.sin(self.desired_action[1])
        self.pub_goal_path_marker.publish(marker)
        # print marker


def main(args=None):
    rclpy.init(args=args)

    cadrl_ros_dir = FindPackageShare("cadrl_ros").find("cadrl_ros")

    a = network.Actions()
    actions = a.actions
    num_actions = a.num_actions
    nn = network.NetworkVP_rnn(network.Config.DEVICE, "network", num_actions)
    nn.simple_load(cadrl_ros_dir + "/checkpoints/network_01900000")

    veh_name = "tb3_01"
    pref_speed = 0.34
    # pref_speed = 0.3
    veh_data = {
        "goal": np.zeros((2,)),
        "radius": 0.3,
        "pref_speed": pref_speed,
        "kw": 10.0,
        "kp": 1.0,
        "name": "tb3_01",
    }

    print("==================================\ncadrl node started")
    print("tb3 speed:", pref_speed, "\n==================================")
    time.sleep(5)
    nn_tb3 = NN_tb3(veh_name, veh_data, nn, actions)

    rclpy.spin(nn_tb3)
    nn_tb3.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
