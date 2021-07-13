<template>
  <div
    class="flowchart-container"
    @mousemove="handleMove"
    @mouseup="handleUp"
    @mousedown="handleDown"
  >
    <svg width="100%" :height="`${height}px`">
      <flowchart-link
        v-bind.sync="link"
        v-for="(link, index) in lines"
        :key="`link${index}`"
        @deleteLink="linkDelete(link.id)"
      ></flowchart-link>
    </svg>
    <flowchart-node
      v-bind.sync="node"
      v-for="(node, index) in scene.layers"
      :key="`node${index}`"
      :options="nodeOptions"
      @linkingStart="linkingStart(node.id)"
      @linkingStop="linkingStop(node.id)"
      @nodeSelected="nodeSelected(node.id, $event)"
    >
    </flowchart-node>
  </div>
</template>

<script>
import FlowchartLink from "./FlowchartLink.vue";
import FlowchartNode from "./FlowchartNode.vue";
import { getMousePosition } from "@/assets/utilty/position";

export default {
  name: "VueFlowchart",
  props: {
    height: {
      type: Number,
      default: 1000,
    },
  },
  data() {
    return {
      action: {
        linking: false,
        dragging: false,
        scrolling: false,
        selected: 0,
      },
      mouse: {
        x: 0,
        y: 0,
        lastX: 0,
        lastY: 0,
      },
      draggingLink: null,
      rootDivOffset: {
        top: 0,
        left: 0,
      },
    };
  },
  components: {
    FlowchartLink,
    FlowchartNode,
  },
  computed: {
    scene: {
      set(val) {
        this.$store.dispatch("data/setData", val);
      },
      get() {
        return this.$store.getters["data/getData"];
      },
    },
    nodeOptions() {
      return {
        centerY: this.scene.centerY,
        centerX: this.scene.centerX,
        scale: this.scene.scale,
        offsetTop: this.rootDivOffset.top,
        offsetLeft: this.rootDivOffset.left,
        selected: this.action.selected,
      };
    },
    lines() {
      const lines = this.scene.links.map((link) => {
        const fromNode = this.findNodeWithID(link.from);
        const toNode = this.findNodeWithID(link.to);
        let x, y, cy, cx, ex, ey;
        x = fromNode.position[0];
        y = fromNode.position[1];
        [cx, cy] = this.getPortPosition("bottom", fromNode, x, y);
        x = toNode.position[0];
        y = toNode.position[1];
        [ex, ey] = this.getPortPosition("top", toNode, x, y);
        return {
          start: [cx, cy],
          end: [ex, ey],
          id: link.id,
        };
      });
      if (this.draggingLink) {
        let x, y, cy, cx;
        const fromNode = this.findNodeWithID(this.draggingLink.from);
        x = fromNode.position[0];
        y = fromNode.position[1];
        [cx, cy] = this.getPortPosition("bottom", fromNode, x, y);
        // push temp dragging link, mouse cursor postion = link end postion
        lines.push({
          start: [cx, cy],
          end: [this.draggingLink.mx, this.draggingLink.my],
        });
      }
      return lines;
    },
  },
  mounted() {
    this.rootDivOffset.top = this.$el ? this.$el.offsetTop : 0;
    this.rootDivOffset.left = this.$el ? this.$el.offsetLeft : 0;
    // console.log(22222, this.rootDivOffset);
  },
  methods: {
    findNodeWithID(id) {
      return this.scene.layers.find((item) => {
        return item.id === id
      })
    },
    getPortPosition(type, layer, x, y) {
      let fontSize = 11
      let nodeMid = ((layer.name.length + layer.type.length)*fontSize)/2;
      if (type === "top") {
        return [x + nodeMid, y];
      } else if (type === "bottom") {
        return [x + nodeMid, y + 40];
      }
    },
    linkingStart(id) {
      this.action.linking = true;
      this.draggingLink = {
        from: id,
        mx: 0,
        my: 0,
      };
    },
    linkingStop(id) {
      // add new Link
      if (this.draggingLink && this.draggingLink.from !== id) {
        // check link existence
        const existed = this.scene.links.find((link) => {
          return link.from === this.draggingLink.from && link.to === id;
        });
        if (!existed) {
          let maxID = Math.max(
            0,
            ...this.scene.links.map((link) => {
              return link.id;
            })
          );
          const newLink = {
            id: maxID + 1,
            from: this.draggingLink.from,
            to: id,
          };
          this.scene.links.push(newLink);
          this.$emit("linkAdded", newLink);
        }
      }
      this.draggingLink = null;
    },
    linkDelete(id) {
      const deletedLink = this.scene.links.find((item) => {
        return item.id === id;
      });
      if (deletedLink) {
        this.scene.links = this.scene.links.filter((item) => {
          return item.id !== id;
        });
        this.$emit("linkBreak", deletedLink);
      }
    },
    nodeSelected(id, e) {
      this.action.dragging = id;
      this.action.selected = id;
      this.$emit("nodeClick", id);
      this.mouse.lastX =
        e.pageX || e.clientX + document.documentElement.scrollLeft;
      this.mouse.lastY =
        e.pageY || e.clientY + document.documentElement.scrollTop;
    },
    handleMove(e) {
      if (this.action.linking) {
        [this.mouse.x, this.mouse.y] = getMousePosition(this.$el, e);
        [this.draggingLink.mx, this.draggingLink.my] = [
          this.mouse.x,
          this.mouse.y,
        ];
      }
      if (Number.isInteger(this.action.dragging)) {
        this.mouse.x =
          e.pageX || e.clientX + document.documentElement.scrollLeft;
        this.mouse.y =
          e.pageY || e.clientY + document.documentElement.scrollTop;
        let diffX = this.mouse.x - this.mouse.lastX;
        let diffY = this.mouse.y - this.mouse.lastY;

        this.mouse.lastX = this.mouse.x;
        this.mouse.lastY = this.mouse.y;
        this.moveSelectedNode(diffX, diffY);
      }
      if (this.action.scrolling) {
        [this.mouse.x, this.mouse.y] = getMousePosition(this.$el, e);
        let diffX = this.mouse.x - this.mouse.lastX;
        let diffY = this.mouse.y - this.mouse.lastY;

        this.mouse.lastX = this.mouse.x;
        this.mouse.lastY = this.mouse.y;

        this.scene.centerX += diffX;
        this.scene.centerY += diffY;

        // this.hasDragged = true
      }
    },
    handleUp(e) {
      const target = e.target || e.srcElement;
      if (this.$el.contains(target)) {
        if (
          typeof target.className !== "string" ||
          target.className.indexOf("node-input") < 0
        ) {
          this.draggingLink = null;
        }
        if (
          typeof target.className === "string" &&
          target.className.indexOf("node-delete") > -1
        ) {
          // console.log('delete2', this.action.dragging);
          this.nodeDelete(this.action.dragging);
        }
      }
      this.action.linking = false;
      this.action.dragging = null;
      this.action.scrolling = false;
    },
    handleDown(e) {
      const target = e.target || e.srcElement;
      // console.log('for scroll', target, e.keyCode, e.which)
      if (
        (target === this.$el || target.matches("svg, svg *")) &&
        e.which === 1
      ) {
        this.action.scrolling = true;
        [this.mouse.lastX, this.mouse.lastY] = getMousePosition(this.$el, e);
        this.action.selected = null; // deselectAll
      }
      this.$emit("canvasClick", e);
    },
    moveSelectedNode(dx, dy) {
      let layer = this.scene.layers.find((item) => {
        return item.id === this.action.dragging;
      })
      let index = this.scene.layers.indexOf(layer);
      let left = this.scene.layers[index].position[0] + dx / this.scene.scale;
      let top = this.scene.layers[index].position[1] + dy / this.scene.scale;
      this.$set(
        this.scene.layers,
        index,
        Object.assign(this.scene.layers[index], { position: [left, top] })
      );
    },
    nodeDelete(id) {
      this.scene.layers = this.scene.layers.filter((node) => {
        return node.id !== id;
      });
      this.scene.links = this.scene.links.filter((link) => {
        return link.from !== id && link.to !== id;
      });
      this.$emit("nodeDelete", id);
    },
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
.flowchart-container {
  margin: 0;
  background: #17212b;
  //17212B
  position: relative;
  overflow: hidden;
  svg {
    cursor: grab;
  }
}
</style>
