import rclpy
from rclpy.node import Node
from ford_msgs.msg import Clusters
from gazebo_msgs.msg import ModelStates
from pedsim_msgs.msg import AgentStates


class Transformer(Node):
    def __init__(self):
        super().__init__("ped_to_ford")

        # Initialize the subscriber and publisher
        self.pedsim_sub = self.create_subscription(
            AgentStates, "/pedsim_simulator/simulated_agents", self.trans_ped_msg, 10
        )

        # Uncomment the following lines if you need to use the Gazebo model states instead
        # self.actors_sub = self.create_subscription(
        #     ModelStates,
        #     '/gazebo/model_states',
        #     self.trans_gaz_to_cluster,
        #     10
        # )

        self.clusters_pub = self.create_publisher(Clusters, "/clusters", 10)

    def trans_ped_msg(self, msg: AgentStates):
        clusters = Clusters()
        mean_points = []
        velocities = []
        labels = []
        for actor in msg.agent_states:
            labels.append(actor.id)
            mean_points.append(actor.pose.position)
            velocities.append(actor.twist.linear)

        clusters.mean_points = mean_points
        clusters.velocities = velocities
        clusters.labels = labels
        self.clusters_pub.publish(clusters)

    def trans_gaz_to_cluster(self, msg: ModelStates):
        clusters = Clusters()
        mean_points = []
        velocities = []
        labels = []

        actors = [
            msg.name.index(name) for name in msg.name if name.startswith("person_")
        ]
        for index in actors:
            mean_points.append(msg.pose[index].position)
            velocities.append(msg.twist[index].linear)
            labels.append(index)

        clusters.mean_points = mean_points
        clusters.velocities = velocities
        clusters.labels = labels
        self.clusters_pub.publish(clusters)


def main(args=None):
    rclpy.init(args=args)
    transformer = Transformer()
    rclpy.spin(transformer)

    # Cleanup
    transformer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
