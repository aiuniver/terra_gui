<template>
  <div class="blocks" :key="key" @contextmenu="contextmenu">
    <net class="blocks__center" :x="centerX" :y="centerY" :scale="scale" />
    <Link class="blocks__lines" :lines="lines" />
    <Block
      v-for="block in blocks"
      :key="block.id"
      :ref="'block_' + block.id"
      v-bind="block"
      :options="optionsForChild"
      :linkingCheck="tempLink"
      @linkingStart="linkingStart(block, $event)"
      @linkingStop="linkingStop(block, $event)"
      @linkingBreak="linkingBreak(block, $event)"
      @select="blockSelect(block)"
      @position="position($event)"
    />
    <!-- <Menu v-bind="menu" :select="selectLength" @click="event" /> -->
  </div>
</template>

<script>
import { mapActions, mapGetters } from 'vuex';
import { mouseHelper } from '@/utils/blocks/utils';
import Block from '@/components/global/overflow/Block';
import Link from '@/components/global/overflow/Link';
// import Net from '@/components/global/overflow/Net';
// import Menu from '@/components/global/overflow/ContextMenu';

export default {
  name: 'Blocks',
  components: {
    Block,
    Link,
    // Net,
    // Menu,
  },
  props: {},
  data: () => ({
    key: 1,
    save: [],
    menu: {},
    dragging: false,
    centerX: 0,
    centerY: 0,
    scale: 1,
    tempLink: null,
    hasDragged: false,
    mouseX: 0,
    mouseY: 0,
    lastMouseX: 0,
    lastMouseY: 0,
    minScale: 0.2,
    maxScale: 5,
    linking: false,
    linkStart: null,
    inputSlotClassName: 'inputSlot',
  }),
  computed: {
    ...mapGetters('create', {
      getKeyEvent: 'getKeyEvent',
      blocks: 'getBlocks',
      links: 'getLinks',
      selectLength: 'getSelectedLength',
    }),
    keyEvent: {
      set(value) {
        this.setKeyEvent(value);
      },
      get() {
        return this.getKeyEvent;
      },
    },
    optionsForChild() {
      return {
        scale: this.scale,
        x: this.centerX,
        y: this.centerY,
      };
    },
    container() {
      return {
        centerX: this.centerX,
        centerY: this.centerY,
        scale: this.scale,
      };
    },
    // Links calculate
    lines() {
      let lines = [];
      for (let link of this.links) {
        let originBlock = this.blocks.find(({ id }) => id === link.originID);
        let targetBlock = this.blocks.find(({ id }) => id === link.targetID);
        if (!originBlock || !targetBlock || originBlock.id === targetBlock.id) {
          console.warn('Remove invalid link', link);
          this.removeLink(link.id);
          continue;
        }
        let originLinkPos = this.getConnectionPos(originBlock, link.originSlot, false);
        let targetLinkPos = this.getConnectionPos(targetBlock, link.targetSlot, true);
        if (!originLinkPos || !targetLinkPos) {
          console.log('Remove invalid link (slot not exist)', link);
          this.removeLink(link.id);
          continue;
        }
        let { x: x1, y: y1 } = originLinkPos;
        let { x: x2, y: y2 } = targetLinkPos;

        lines.push({
          x1,
          y1,
          x2,
          y2,
          slot: link.originSlot,
          scale: this.scale,
          style: {
            stroke: 'rgb(101, 124, 244)',
            strokeWidth: 2 * this.scale,
            fill: 'none',
            zIndex: 999,
          },
          outlineStyle: {
            stroke: '#666',
            strokeWidth: 2 * this.scale,
            strokeOpacity: 0.6,
            fill: 'none',
            zIndex: 999,
          },
        });
      }
      if (this.tempLink) {
        lines.push(this.tempLink);
      }
      return lines;
    },
  },
  methods: {
    ...mapActions('create', [
      'add',
      'cloneAll',
      'align',
      'distance',
      'remove',
      'clone',
      'select',
      'setKeyEvent',
      'deselect',
      'position',
      'update',
      'addLink',
      'updateLink',
      'removeLink',
    ]),
    event(value) {
      this.menu = {};
      console.log(value);
      if (value === 'add') this.add({});
      if (value === 'delete') this.remove();
      if (value === 'clone') this.cloneAll();
      if (value === 'left') this.align('ArrowLeft');
      if (value === 'right') this.align('ArrowRight');
      if (value === 'up') this.align('ArrowUp');
      if (value === 'down') this.align('ArrowDown');
      if (value === 'center') this.align('center');
      if (value === 'vertical') this.distance('vertical');
      if (value === 'horizon') this.distance('horizon');
      if (value === 'select') this.deselect(true);
      this.key += 1;
    },
    contextmenu(e) {
      // e.preventDefault();
      console.log(e.clientX, e.clientY);
      // this.menu = { x: e.clientX, y: e.clientY };
    },
    handleMauseOver(e) {
      this.mouseIsOver = e.type === 'mouseenter';
    },
    keyup(event) {
      this.keyEvent = event;
      const { code, ctrlKey, shiftKey } = event;
      const mouseIsOver = this.mouseIsOver;
      console.log(event);
      if (event.type === 'keyup') {
        if (mouseIsOver && code === 'Delete') {
          this.remove();
        }
        if (mouseIsOver && code === 'KeyA' && ctrlKey) {
          this.deselect(true);
        }
        if (mouseIsOver && ['ArrowLeft', 'ArrowRight', 'ArrowDown', 'ArrowUp'].includes(code) && ctrlKey && !shiftKey) {
          this.align(code);
          this.key += 1;
        }

        if (mouseIsOver && ['ArrowUp'].includes(code) && ctrlKey && shiftKey) {
          this.distance('vertical');
          this.key += 1;
        }
        if (mouseIsOver && code === 'KeyC' && ctrlKey) {
          this.save = this.blocks.filter(block => block.selected);
        }
        if (mouseIsOver && code === 'KeyX' && ctrlKey) {
          this.save = JSON.parse(JSON.stringify(this.blocks.filter(block => block.selected)));
          console.log(this.save);
          this.remove();
        }
        if (mouseIsOver && code === 'KeyV' && ctrlKey) {
          this.deselect();
          this.save.forEach(block => {
            this.clone(block);
          });
        }
      }
    },
    handleMove(e) {
      let mouse = mouseHelper(this.$el, e);
      this.mouseX = mouse.x;
      this.mouseY = mouse.y;
      if (this.dragging) {
        console.log('handleMove');
        let diffX = this.mouseX - this.lastMouseX;
        let diffY = this.mouseY - this.lastMouseY;
        this.lastMouseX = this.mouseX;
        this.lastMouseY = this.mouseY;
        this.centerX += diffX;
        this.centerY += diffY;
        this.hasDragged = true;
      }
      if (this.linking && this.linkStart) {
        let linkStartPos = this.getConnectionPos(this.linkStart.block, this.linkStart.slot, false);
        this.tempLink = {
          x1: linkStartPos.x,
          y1: linkStartPos.y,
          x2: this.mouseX,
          y2: this.mouseY,
          slot: this.linkStart.slot,
          style: {
            stroke: '#8f8f8f',
            strokeWidth: 2 * this.scale,
            fill: 'none',
          },
        };
      }
    },
    handleDown(e) {
      console.log('handleDown');
      console.log(e);
      const target = e.target || e.srcElement;
      if ((target === this.$el || target.matches('svg, svg *')) && e.which === 1) {
        this.dragging = true;
        let mouse = mouseHelper(this.$el, e);
        this.mouseX = mouse.x;
        this.mouseY = mouse.y;
        this.lastMouseX = this.mouseX;
        this.lastMouseY = this.mouseY;
        if (!this.keyEvent?.ctrlKey) {
          this.deselect();
        }
        if (e.preventDefault) e.preventDefault();
      }
    },
    handleUp(e) {
      console.log('handleUp');
      const target = e.target || e.srcElement;
      if (this.dragging) {
        this.dragging = false;
        if (this.hasDragged) {
          this.hasDragged = false;
        }
      }
      if (this.$el.contains(target) && (typeof target.className !== 'string' || !target.className.includes(this.inputSlotClassName))) {
        this.linking = false;
        this.tempLink = null;
        this.linkStart = null;
      }
    },
    handleWheel(e) {
      const target = e.target || e.srcElement;
      if (this.$el.contains(target)) {
        let deltaScale = Math.pow(1.1, e.deltaY * -0.01);
        this.scale *= deltaScale;
        if (this.scale < this.minScale) {
          this.scale = this.minScale;
          return;
        } else if (this.scale > this.maxScale) {
          this.scale = this.maxScale;
          return;
        }
        let deltaOffsetX = (this.mouseX - this.centerX) * (deltaScale - 1);
        let deltaOffsetY = (this.mouseY - this.centerY) * (deltaScale - 1);
        this.centerX -= deltaOffsetX;
        this.centerY -= deltaOffsetY;
      }
    },
    getConnectionPos(block, slot, isInput) {
      if (!block || slot === -1) {
        return undefined;
      }
      let x = 0;
      let y = 0;
      x += block.position[0];
      y += block.position[1];
      const { width = 0, heigth = 0 } = this.$refs?.['block_' + block.id]?.[0]?.getH() || {};
      if (isInput) {
        x += width / 2;
        y += -3;
      } else {
        x += width / 2;
        y += heigth;
      }

      x *= this.scale;
      y *= this.scale;

      x += this.centerX;
      y += this.centerY;

      return { x, y };
    },
    linkingStart(block, slot) {
      console.log('linkingStart');
      this.linkStart = { block, slot };
      let linkStartPos = this.getConnectionPos(block, slot, false);
      this.tempLink = {
        x1: linkStartPos.x,
        y1: linkStartPos.y,
        x2: this.mouseX,
        y2: this.mouseY,
        style: {
          stroke: '#8f8f8f',
          strokeWidth: 2 * this.scale,
          fill: 'none',
        },
      };
      this.linking = true;
    },
    linkingStop(target, slot) {
      console.log('linkingStop');
      if (this.linkStart && target && slot > -1) {
        const {
          slot: originSlot,
          block: { id: originID },
        } = this.linkStart;
        const targetID = target.id;
        const targetSlot = slot;
        const links = this.links.filter(line => {
          return (
            !(line.targetID === targetID && line.targetSlot === targetSlot && line.originID === originID && line.originSlot === originSlot) &&
            !((line.targetID === originID && line.originID === targetID) || (line.originID === originID && line.targetID === targetID))
          );
        });
        this.updateLink(links);

        let maxID = Math.max(0, ...this.links.map(o => o.id));
        if (this.linkStart.block.id !== target.id) {
          const originID = this.linkStart.block.id;
          const originSlot = this.linkStart.slot;
          const targetID = target.id;
          const targetSlot = slot;

          this.addLink({
            id: maxID + 1,
            originID,
            originSlot,
            targetID,
            targetSlot,
          });
          this.updateModel();
        }
      }

      this.linking = false;
      this.tempLink = null;
      this.linkStart = null;
    },
    linkingBreak(target, slot) {
      console.log('linkingBreak');
      if (target && slot > -1) {
        let findLink = this.links.find(({ targetID, targetSlot }) => targetID === target.id && targetSlot === slot);
        if (findLink) {
          let findBlock = this.blocks.find(({ id }) => id === findLink.originID);
          const links = this.links.filter(({ targetID, targetSlot }) => !(targetID === target.id && targetSlot === slot));
          this.updateLink(links);
          this.linkingStart(findBlock, findLink.originSlot);
          this.updateModel();
        }
      }
    },

    position({ left, top }) {
      if (!this.keyEvent.ctrlKey) {
        const update = this.blocks.map(b => {
          const [x, y] = b.position;
          const position = b.selected ? [x + left, y + top] : [x, y];
          return { ...b, position };
        });
        this.update(update);
      }
    },

    blockSelect({ id, selected }) {
      console.log(id, selected);
      if (!selected || this.keyEvent.ctrlKey) {
        this.select({ id });
      }
    },

    updateModel() {
      // this.$store.dispatch("blocks/updateModel");
    },
  },

  mounted() {
    const doc = document.documentElement;
    this.$el.addEventListener('mouseenter', this.handleMauseOver);
    this.$el.addEventListener('mouseleave', this.handleMauseOver);
    doc.addEventListener('keydown', this.keyup);
    doc.addEventListener('keyup', this.keyup);
    doc.addEventListener('mousemove', this.handleMove, true);
    doc.addEventListener('mousedown', this.handleDown, true);
    doc.addEventListener('mouseup', this.handleUp, true);
    doc.addEventListener('wheel', this.handleWheel, true);
    this.centerX = this.$el.clientWidth / 2;
    this.centerY = this.$el.clientHeight / 2;
  },
  beforeDestroy() {
    const doc = document.documentElement;
    this.$el.removeEventListener('mouseenter', this.handleMauseOver);
    this.$el.removeEventListener('mouseleave', this.handleMauseOver);
    doc.removeEventListener('keydown', this.keyup);
    doc.removeEventListener('keyup', this.keyup);
    doc.removeEventListener('mousemove', this.handleMove, true);
    doc.removeEventListener('mousedown', this.handleDown, true);
    doc.removeEventListener('mouseup', this.handleUp, true);
    doc.removeEventListener('wheel', this.handleWheel, true);
  },
};
</script>

<style lang="scss">
.blocks {
  height: 100%;
  width: 100%;
  // background-color: #17212b;
  position: relative;
  overflow: hidden;
  &__lines {
    position: absolute;
  }
  &__center {
    position: absolute;
  }
}
</style>
