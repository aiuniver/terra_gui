<template>
  <div class="files-menu" :class="{ 'files-menu-root': isRoot }">
    <div class="files-menu-nodes-list">
      <div
        v-for="(node, nodeInd) in nodes"
        :class="['files-menu-node', { 'files-menu-selected': node.isSelected }]"
        :key="'nodeInd_' + nodeInd"
      >
        <div
          class="files-menu-node-item"
          :draggable="node.dragndrop"
          @dragstart="dragstart($event, node)"
          @dragover.stop
          :style="!node.dragndrop ? `opacity: 0.5;` : null"
        >
          <div v-for="(gapInd, i) in gaps" class="files-menu-gap" :key="'gap_' + i"></div>

          <div class="files-menu__title" @click="onToggleHandler($event, node)">
            <span
              v-if="node.children.length"
              :class="['icons files-menu__title--toggle', { rotate: !node.isExpanded }]"
            />
            <span v-else class="files-menu__title--empty" />
            <span v-if="node.children.length" class="icons files-menu__title--folder" />
            <span v-else class="icons files-menu__title--file" :class="getFileTypeClass(node)" />
            <span
              class="files-menu__title--text"
              v-text="node.title"
              :style="{ cursor: node.type === 'folder' ? 'grab' : '', flex: '1' }"
            />
          </div>
        </div>

        <files-menu
          v-if="node.children && node.children.length && !node.isExpanded"
          :value="node.children"
          :level="node.level"
          :parentInd="nodeInd"
          :allowMultiselect="allowMultiselect"
          :allowToggleBranch="allowToggleBranch"
          :edgeSize="edgeSize"
          :showBranches="showBranches"
          @dragover.prevent
        ></files-menu>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'files-menu',
  props: {
    value: {
      type: Array,
      default: () => [],
    },
    edgeSize: {
      type: Number,
      default: 3,
    },
    showBranches: {
      type: Boolean,
      default: false,
    },
    level: {
      type: Number,
      default: 0,
    },
    parentInd: {
      type: Number,
    },
    allowMultiselect: {
      type: Boolean,
      default: true,
    },
    allowToggleBranch: {
      type: Boolean,
      default: true,
    },
    path: String,
    dragndrop: Boolean,
    cover: String,
  },
  data() {
    return {
      currentValue: this.value,
    };
  },

  watch: {
    value: function (newValue) {
      this.currentValue = newValue;
    },
  },

  computed: {
    nodes() {
      if (this.isRoot) {
        const nodeModels = this.copy(this.currentValue);
        return this.getNodes(nodeModels);
      }

      return this.getParent().nodes[this.parentInd].children;
    },
    gaps() {
      const gaps = [];
      let i = this.level - 1;
      if (!this.showBranches) i++;
      while (i-- > 0) gaps.push(i);
      return gaps;
    },
    isRoot() {
      return !this.level;
    },
  },
  methods: {
    getFileTypeClass(node) {
      return {
        [`icon-file-${node.type}`]: true,
      };
    },
    dragstart({ dataTransfer }, { path, title, type, cover, table }) {
      // var img = document.createElement('img');
      // img.src = 'http://kr.org/images/hacker.png';
      // dataTransfer.setDragImage(img, 0, 0);
      dataTransfer.setData('CardDataType', JSON.stringify({ value: path, label: title, type, id: 0, cover, table }));
      dataTransfer.effectAllowed = 'move';
    },

    getNodes(nodeModels, parentPath = [], isVisible = true) {
      return nodeModels.map((nodeModel, ind) => {
        const nodePath = parentPath.concat(ind);
        return this.getNode(nodePath, nodeModel, nodeModels, isVisible);
      });
    },

    getNode(fpath, nodeModel = null, siblings = null, isVisible = null) {
      const ind = fpath.slice(-1)[0];

      // calculate nodeModel, siblings, isVisible fields if it is not passed as arguments
      siblings = siblings || this.getNodeSiblings(this.currentValue, fpath);
      nodeModel = nodeModel || (siblings && siblings[ind]) || null;

      if (isVisible == null) {
        isVisible = this.isVisible(fpath);
      }

      if (!nodeModel) return null;

      const isExpanded = nodeModel.isExpanded == void 0 ? true : !!nodeModel.isExpanded;
      // const isDraggable =
      //   nodeModel.isDraggable == void 0 ? true : !!nodeModel.isDraggable;
      // const isSelectable =
      //   nodeModel.isSelectable == void 0 ? true : !!nodeModel.isSelectable;

      const node = {
        // define the all ISlTreeNodeModel props
        title: nodeModel.title,
        path: nodeModel.path,
        type: nodeModel.type,
        dragndrop: nodeModel.dragndrop,
        cover: nodeModel.cover,
        table: nodeModel.data,
        isLeaf: !!nodeModel.isLeaf,
        children: nodeModel.children ? this.getNodes(nodeModel.children, fpath, isExpanded) : [],
        isSelected: !!nodeModel.isSelected,
        isExpanded,
        isVisible,
        // isDraggable,
        // isSelectable,
        data: nodeModel.data !== void 0 ? nodeModel.data : {},

        // define the all ISlTreeNode computed props
        fpath: fpath,
        pathStr: JSON.stringify(fpath),
        level: fpath.length,
      };
      return node;
    },

    isVisible(fpath) {
      if (fpath.length < 2) return true;
      let nodeModels = this.currentValue;

      for (let i = 0; i < fpath.length - 1; i++) {
        let ind = fpath[i];
        let nodeModel = nodeModels[ind];
        let isExpanded = nodeModel.isExpanded == void 0 ? true : !!nodeModel.isExpanded;
        if (!isExpanded) return false;
        nodeModels = nodeModel.children;
      }

      return true;
    },

    emitInput(newValue) {
      this.currentValue = newValue;
      this.getRoot().$emit('input', newValue);
    },

    onToggleHandler(event, node) {
      if (!this.allowToggleBranch) return;
      this.updateNode(node.fpath, { isExpanded: !node.isExpanded });
      event.stopPropagation();
    },

    getParent() {
      return this.$parent;
    },

    getRoot() {
      if (this.isRoot) return this;
      return this.getParent().getRoot();
    },

    updateNode(fpath, patch) {
      if (!this.isRoot) {
        this.getParent().updateNode(fpath, patch);
        return;
      }

      const pathStr = JSON.stringify(fpath);
      const newNodes = this.copy(this.currentValue);
      this.traverse((node, nodeModel) => {
        if (node.pathStr !== pathStr) return;
        Object.assign(nodeModel, patch);
      }, newNodes);

      this.emitInput(newNodes);
    },

    traverse(cb, nodeModels = null, parentPath = []) {
      if (!nodeModels) nodeModels = this.currentValue;

      let shouldStop = false;

      const nodes = [];

      for (let nodeInd = 0; nodeInd < nodeModels.length; nodeInd++) {
        const nodeModel = nodeModels[nodeInd];
        const itemPath = parentPath.concat(nodeInd);
        const node = this.getNode(itemPath, nodeModel, nodeModels);
        shouldStop = cb(node, nodeModel, nodeModels) === false;
        nodes.push(node);

        if (shouldStop) break;

        if (nodeModel.children) {
          shouldStop = this.traverse(cb, nodeModel.children, itemPath) === false;
          if (shouldStop) break;
        }
      }

      return !shouldStop ? nodes : false;
    },

    traverseModels(cb, nodeModels) {
      let i = nodeModels.length;
      while (i--) {
        const nodeModel = nodeModels[i];
        if (nodeModel.children) this.traverseModels(cb, nodeModel.children);
        cb(nodeModel, nodeModels, i);
      }
      return nodeModels;
    },
    copy(entity) {
      return JSON.parse(JSON.stringify(entity));
    },
  },
};
</script>

<style lang="scss" scoped>
.files-menu {
  position: relative;
  cursor: default;
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  &.files-menu-root {
    color: rgba(255, 255, 255, 0.5);
    border-radius: 3px;
  }
  &__title {
    display: flex;
    font-family: Open Sans;
    font-style: normal;
    font-weight: normal;
    font-size: 14px;
    line-height: 17px;
    color: #ffffff;
    &--toggle {
      width: 15px;
      height: 15px;
      margin-right: 7px;
      background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iOCIgaGVpZ2h0PSIxMiIgdmlld0JveD0iMCAwIDggMTIiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xLjI5IDAuNzA5OTk5QzAuODk5OTk4IDEuMSAwLjg5OTk5OCAxLjczIDEuMjkgMi4xMkw1LjE3IDZMMS4yOSA5Ljg4QzAuODk5OTk4IDEwLjI3IDAuODk5OTk4IDEwLjkgMS4yOSAxMS4yOUMxLjY4IDExLjY4IDIuMzEgMTEuNjggMi43IDExLjI5TDcuMjkgNi43QzcuNjggNi4zMSA3LjY4IDUuNjggNy4yOSA1LjI5TDIuNyAwLjY5OTk5OUMyLjMyIDAuMzE5OTk5IDEuNjggMC4zMTk5OTkgMS4yOSAwLjcwOTk5OVoiIGZpbGw9IiNBN0JFRDMiLz4KPC9zdmc+Cg==);
      &.rotate {
        transform: rotate(90deg);
      }
    }
    &--empty {
      width: 15px;
      height: 15px;
      margin-right: 7px;
    }
    &--folder {
      width: 20px;
      height: 16px;
      margin-right: 7px;
      background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAyMCAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTE4IDJIMTBMOC41OSAwLjU5QzguMjEgMC4yMSA3LjcgMCA3LjE3IDBIMkMwLjkgMCAwLjAwOTk5OTk5IDAuOSAwLjAwOTk5OTk5IDJMMCAxNEMwIDE1LjEgMC45IDE2IDIgMTZIMThDMTkuMSAxNiAyMCAxNS4xIDIwIDE0VjRDMjAgMi45IDE5LjEgMiAxOCAyWk0xNyAxNEgzQzIuNDUgMTQgMiAxMy41NSAyIDEzVjVDMiA0LjQ1IDIuNDUgNCAzIDRIMTdDMTcuNTUgNCAxOCA0LjQ1IDE4IDVWMTNDMTggMTMuNTUgMTcuNTUgMTQgMTcgMTRaIiBmaWxsPSIjQTdCRUQzIi8+Cjwvc3ZnPgo=);
    }
    &--file {
      width: 18px;
      height: 18px;
      margin-right: 8px;
      background-size: contain;
    }
  }
}

.icons {
  display: inline-block;
  background-position: center;
  background-repeat: no-repeat;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.files-menu-root > .files-menu-nodes-list {
  overflow: hidden;
  position: relative;
  padding-bottom: 4px;
}

.files-menu-selected > .files-menu-node-item {
  background-color: #13242d;
  color: white;
}

.files-menu-node-item {
  &:hover,
  &.files-menu-cursor-hover {
    color: white;
  }

  position: relative;
  display: flex;
  flex-direction: row;
  padding: 4px 10px;
  line-height: 28px;
  border: 1px solid transparent;

  &.files-menu-cursor-inside {
    border: 1px solid rgba(255, 255, 255, 0.5);
  }
}

.files-menu-gap {
  width: 25px;
  min-height: 1px;
}

.files-menu-sidebar {
  margin-left: auto;
}

.files-menu-cursor {
  position: absolute;
  border: 1px solid rgba(255, 255, 255, 0.5);
  height: 1px;
  width: 100%;
}

.files-menu-drag-info {
  position: absolute;
  background-color: rgba(0, 0, 0, 0.5);
  opacity: 0.5;
  margin-left: 20px;
  padding: 5px 10px;
}
</style>
