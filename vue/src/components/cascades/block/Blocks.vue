<template>
  <div class="t-block" @contextmenu="contextmenu">
    <Net class="t-block__center" :x="centerX" :y="centerY" :scale="scale" />
    <VueLink class="t-block__lines" :lines="lines" />
    <VueBlock
      v-for="block in blocks"
      :key="block.id"
      :ref="'block_' + block.id"
      v-bind="block"
      :options="optionsForChild"
      :linkingCheck="tempLink"
      :icons="icons"
      :filter="filter"
      :errors="errors"
      @linkingStart="linkingStart(block, $event)"
      @linkingStop="linkingStop(block, $event)"
      @linkingBreak="linkingBreak(block, $event)"
      @select="blockSelect(block)"
      @position="position(block, $event)"
      @moveBlock="moveBlock"
      @clickIcons="clickIcons($event, block)"
    />
    <div class="btn-zoom">
      <div class="btn-zoom__item">
        <i class="t-icon icon-zoom-inc" @click="zoom(1)"></i>
      </div>
      <hr />
      <div class="btn-zoom__item">
        <i class="t-icon icon-zoom-reset" @click="zoom(0)"></i>
      </div>
      <hr />
      <div class="btn-zoom__item">
        <i class="t-icon icon-zoom-dec" @click="zoom(-1)"></i>
      </div>
    </div>
  </div>
</template>

<script>
import domtoimage from '@/assets/js/dom-to-image.min.js';
import { mouseHelper } from '@/store/const/cascades';
import { mapGetters } from 'vuex';
import VueBlock from './VueBlock';
import VueLink from './VueLink';
import Net from './Net';

export default {
  name: 'VueBlockContainer',
  components: {
    VueBlock,
    VueLink,
    Net,
  },
  props: {
    blocksContent: {
      type: Array,
      default() {
        return [];
      },
    },
    options: {
      type: Object,
    },
  },
  data: () => ({
    icons: [
      { icon: 'icon-modeling-copy-white', event: 'clone' },
      { icon: 'icon-modeling-link-remove', event: 'link' },
      { icon: 'icon-modeling-remove', event: 'remove' },
    ],
    dragging: false,
    //
    centerX: 0,
    centerY: 0,
    scale: 1,
    //
    // blocks: [],
    // links: [],
    //
    tempLink: null,
    selectedBlock: null,
    hasDragged: false,

    mouseX: 0,
    mouseY: 0,
    lastMouseX: 0,
    lastMouseY: 0,
    minScale: 0.2,
    maxScale: 5,
    linking: false,
    linkStartData: null,
    inputSlotClassName: 'inputSlot',

    defaultScene: {
      blocks: [],
      links: [],
      container: {},
    },
  }),

  computed: {
    ...mapGetters({
      project: 'projects/getProject',
    }),
    scaleCenter() {
      return {
        top: this.centerY + 'px',
        left: this.centerX + 'px',
        transform: 'scale(' + (this.scale + '') + ')',
        transformOrigin: 'top left',
      };
    },
    filter() {
      return {
        InputData: this.blocks.filter(i => i.group === 'InputData').length > 1 ? ['clone', 'link', 'remove'] : ['link'],
        Model: ['clone', 'link', 'remove'],
        Function: ['clone', 'link', 'remove'],
        Custom: ['clone', 'link', 'remove'],
        Service: ['clone', 'link', 'remove'],
        OutputData: this.blocks.filter(i => i.group === 'OutputData').length > 1 ? ['clone', 'link', 'remove'] : ['link'],
      };
    },
    errors() {
      return this.$store.getters['cascades/getErrorsBlocks'];
    },
    blocks: {
      set(value) {
        this.$store.dispatch('cascades/setBlocks', value);
      },
      get() {
        return this.$store.getters['cascades/getBlocks'];
      },
    },
    block() {
      return this.$store.getters['cascades/getBlock'];
    },
    links: {
      set(value) {
        console.log(value);
        this.$store.dispatch('cascades/setLinks', value);
      },
      get() {
        return this.$store.getters['cascades/getLinks'];
      },
    },
    optionsForChild() {
      // console.log(this.centerX, this.centerY);
      return {
        width: 180,
        titleHeight: 42,
        scale: this.scale,
        inputSlotClassName: this.inputSlotClassName,
        center: {
          x: this.centerX,
          y: this.centerY,
        },
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
        let originBlock = this.blocks.find(block => {
          return block.id === link.originID;
        });

        let targetBlock = this.blocks.find(block => {
          return block.id === link.targetID;
        });

        if (!originBlock || !targetBlock) {
          console.log('Remove invalid link', link);
          this.$store.dispatch('cascades/removeLink', link.id);
          continue;
        }

        if (originBlock.id === targetBlock.id) {
          console.log('Loop detected, remove link', link);
          this.$store.dispatch('cascades/removeLink', link.id);
          continue;
        }

        let originLinkPos = this.getConnectionPos(originBlock, link.originSlot, false);
        let targetLinkPos = this.getConnectionPos(targetBlock, link.targetSlot, true);

        if (!originLinkPos || !targetLinkPos) {
          console.log('Remove invalid link (slot not exist)', link);
          this.$store.dispatch('cascades/removeLink', link.id);
          continue;
        }

        let x1 = originLinkPos.x;
        let y1 = originLinkPos.y;

        let x2 = targetLinkPos.x;
        let y2 = targetLinkPos.y;
        const select = originBlock.id === (this.block?.id || -1) || targetBlock.id === (this.block?.id || -1);
        lines.push({
          x1: x1,
          y1: y1,
          x2: x2,
          y2: y2,
          slot: link.originSlot,
          scale: this.scale,
          style: {
            stroke: !select ? '#467ca1' : '#a0d5f9',
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
        // eslint-disable-next-line vue/no-side-effects-in-computed-properties
        this.tempLink.style = {
          // eslint-disable-line
          stroke: '#8f8f8f',
          strokeWidth: 2 * this.scale,
          fill: 'none',
        };

        lines.push(this.tempLink);
      }

      return lines;
    },
  },
  methods: {
    contextmenu(e) {
      if (!this.$config.isDev) {
        e.preventDefault();
      }
    },
    getCenter() {
      if (this.scale > 1.5) {
        this.scale = 1.5;
      }
      const x = this.$el.clientWidth / 2 - this.optionsForChild.width / 2 - this.centerX;
      const y = this.$el.clientHeight / 2 - this.centerY;
      console.log(this.centerX);
      console.log(this.centerY);

      return [x, y];
    },
    getError(id) {
      return this.errorsBlocks?.[id] || '';
    },
    clickIcons({ event }, block) {
      console.log(event);
      if (event === 'remove') {
        this.$store.dispatch('cascades/removeBlock', block);
      }
      if (event === 'clone') {
        this.$store.dispatch('cascades/cloneBlock', block);
      }
      if (event === 'link') {
        this.$store.dispatch('cascades/removeLinkToBlock', block);
      }
    },
    handleMauseOver(e) {
      this.mouseIsOver = e.type === 'mouseenter';
    },
    keyup(event) {
      const { code, ctrlKey } = event;
      const mouseIsOver = this.mouseIsOver;
      console.log(mouseIsOver, code);
      if (mouseIsOver && code === 'Delete') {
        if (this.selectedBlock) {
          // this.blockDelete(this.selectedBlock);
        }
      }
      if (mouseIsOver && code === 'KeyC' && ctrlKey) {
        if (this.selectedBlock) {
          // this.blockDelete(this.selectedBlock);
        }
      }
      // console.log(event)
    },
    zoom(value) {
      if (value === 0) {
        this.scale = 1;
        this.centerX = this.$el.clientWidth / 2;
        this.centerY = this.$el.clientHeight / 2;
        return;
      }

      let deltaScale = value === 1 ? 1.1 : 0.9090909090909091;
      this.scale *= deltaScale;
      // this.scale = (value === 1) ? this.scale + 0.1 : this.scale - 0.1;
      if (this.scale < this.minScale) {
        this.scale = this.minScale;
        return;
      } else if (this.scale > this.maxScale) {
        this.scale = this.maxScale;
        return;
      }
      let deltaOffsetX = (this.$el.clientWidth / 2 - this.centerX) * (deltaScale - 1);
      let deltaOffsetY = (this.$el.clientHeight / 2 - this.centerY) * (deltaScale - 1);

      this.centerX -= deltaOffsetX;
      this.centerY -= deltaOffsetY;

      // this.updateScene();
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

      if (this.linking && this.linkStartData) {
        let linkStartPos = this.getConnectionPos(this.linkStartData.block, this.linkStartData.slotNumber, false);
        this.tempLink = {
          x1: linkStartPos.x,
          y1: linkStartPos.y,
          x2: this.mouseX,
          y2: this.mouseY,
          slot: this.linkStartData.slotNumber,
        };
      }
    },
    handleDown(e) {
      console.log('handleDown');
      const target = e.target || e.srcElement;
      if ((target === this.$el || target.matches('svg, svg *')) && e.which === 1) {
        this.dragging = true;

        let mouse = mouseHelper(this.$el, e);
        this.mouseX = mouse.x;
        this.mouseY = mouse.y;

        this.lastMouseX = this.mouseX;
        this.lastMouseY = this.mouseY;

        this.$store.dispatch('cascades/deselectBlocks');
        if (e.preventDefault) e.preventDefault();
      }
    },
    handleUp(e) {
      console.log('handleUp');
      const target = e.target || e.srcElement;

      if (this.dragging) {
        this.dragging = false;

        if (this.hasDragged) {
          // this.updateScene();
          this.hasDragged = false;
          // console.log('вввввввввввввввв');
        }
      }

      if (
        this.$el.contains(target) &&
        (typeof target.className !== 'string' || !target.className.includes(this.inputSlotClassName))
      ) {
        this.linking = false;
        this.tempLink = null;
        this.linkStartData = null;
      }
    },
    handleWheel(e) {
      const target = e.target || e.srcElement;
      if (this.$el.contains(target)) {
        // if (e.preventDefault) e.preventDefault()

        // console.log(e.deltaY);

        let deltaScale = Math.pow(1.1, e.deltaY * -0.01);
        // console.log(deltaScale);
        this.scale *= deltaScale;

        if (this.scale < this.minScale) {
          this.scale = this.minScale;
          return;
        } else if (this.scale > this.maxScale) {
          this.scale = this.maxScale;
          return;
        }

        // console.log(this.mouseX);
        let zoomingCenter = {
          x: this.mouseX,
          y: this.mouseY,
        };

        let deltaOffsetX = (zoomingCenter.x - this.centerX) * (deltaScale - 1);
        let deltaOffsetY = (zoomingCenter.y - this.centerY) * (deltaScale - 1);

        this.centerX -= deltaOffsetX;
        this.centerY -= deltaOffsetY;

        // console.log(this.centerX, this.centerY);
        // this.updateScene();
      }
    },
    // Processing
    getConnectionPos(block, slotNumber, isInput) {
      if (!block || slotNumber === -1) {
        return undefined;
      }

      let x = 0;
      let y = 0;

      x += block.position[0];
      y += block.position[1];

      if (isInput && block.inputs.length > slotNumber) {
        if (block.inputs.length === 1) {
          x += this.optionsForChild.width / 2;
          y += -3;
        } else {
          x += this.optionsForChild.width / 2 - (block.inputs.length * 10) / 2;
          x += 20 * slotNumber;
        }
      } else if (!isInput && block.outputs.length > slotNumber) {
        if (slotNumber === 0) {
          x += this.optionsForChild.width / 2;
          // console.log()
          // y += this.$refs?.['block_' + block.id]?.[0]?.getHeight();
          y += 55;
        }
        if (slotNumber === 1) {
          x += this.optionsForChild.width;
          y += 25;
        }
        if (slotNumber === 2) {
          y += 25;
        }
      } else {
        console.error('slot ' + slotNumber + ' not found, is input: ' + isInput, block);
        return undefined;
      }

      // (height / 2 + blockBorder + padding)
      // y += (20 / 2 + 1 + 2)
      //  + (height * slotNumber)
      // y += (16 * slotNumber)

      x *= this.scale;
      y *= this.scale;

      x += this.centerX;
      y += this.centerY;

      return { x, y };
    },
    // Linking
    findindexBlock(id) {
      return this.blocks.findIndex(block => {
        return block.id === id;
      });
    },
    linkingStart(block, slotNumber) {
      console.log('linkingStart');
      // block.outputs[slotNumber].active = true
      this.linkStartData = { block, slotNumber };
      let linkStartPos = this.getConnectionPos(this.linkStartData.block, this.linkStartData.slotNumber, false);
      this.tempLink = {
        x1: linkStartPos.x,
        y1: linkStartPos.y,
        x2: this.mouseX,
        y2: this.mouseY,
      };

      this.linking = true;
    },
    linkingStop(targetBlock, slotNumber) {
      console.log('linkingStop');
      console.log(targetBlock);
      console.log(this.linkStartData);
      this.linkStartData.block.id;
      if (this.linkStartData && targetBlock && slotNumber > -1) {
        const {
          slotNumber: originSlot,
          block: { id: originID },
        } = this.linkStartData;
        const targetID = targetBlock.id;
        const targetSlot = slotNumber;
        this.links = this.links.filter(line => {
          return (
            !(
              line.targetID === targetID &&
              line.targetSlot === targetSlot &&
              line.originID === originID &&
              line.originSlot === originSlot
            ) &&
            !(
              (line.targetID === originID && line.originID === targetID) ||
              (line.originID === originID && line.targetID === targetID)
            )
          );
        });

        let maxID = Math.max(0, ...this.links.map(o => o.id));
        if (this.linkStartData.block.id !== targetBlock.id) {
          const originID = this.linkStartData.block.id;
          const originSlot = this.linkStartData.slotNumber;
          const targetID = targetBlock.id;
          const targetSlot = slotNumber;

          this.links.push({
            id: maxID + 1,
            originID,
            originSlot,
            targetID,
            targetSlot,
          });
          // console.log("adddd");
          // console.log(originID);
          // const indexOriginBlock = this.findindexBlock(originID)
          // const indexTargetBlock = this.findindexBlock(targetID)
          // if (!this.blocks[indexOriginBlock].bind.down.includes(targetID)) {
          //   this.blocks[indexOriginBlock].bind.down.push(+targetID)
          // }
          // if (!this.blocks[indexTargetBlock].bind.up.includes(originID)) {
          //   this.blocks[indexTargetBlock].bind.up.push(+originID)
          // }
          this.updateModel();
        }
      }

      this.linking = false;
      this.tempLink = null;
      this.linkStartData = null;
    },
    linkingBreak(targetBlock, slotNumber) {
      console.log('linkingBreak');
      if (targetBlock && slotNumber > -1) {
        let findLink = this.links.find(value => {
          return value.targetID === targetBlock.id && value.targetSlot === slotNumber;
        });

        if (findLink) {
          let findBlock = this.blocks.find(value => {
            return value.id === findLink.originID;
          });

          this.links = this.links.filter(value => {
            return !(value.targetID === targetBlock.id && value.targetSlot === slotNumber);
          });
          targetBlock.inputs[findLink.targetSlot].active = false;
          findBlock.outputs[findLink.originSlot].active = false;
          this.linkingStart(findBlock, findLink.originSlot);

          this.updateModel();
          // this.updateScene();
        }
      }
    },
    // removeLink(linkID) {
    //   console.log('removeLink');
    //   this.links = this.links.filter(value => {
    //     return !(value.id === linkID);
    //   });
    // },
    async getImages() {
      const tags = ['line', 'circle'];
      try {
        const image = await domtoimage.toPng(this.$el, {
          bgcolor: '#00000000',
          filter: node => {
            return !(['btn-zoom'].includes(node.className) || tags.includes(node.tagName));
          },
        });
        return image;
      } catch (error) {
        console.log(error);
        return null;
      }
    },
    // Blocks
    // addCloneBlock(oldBlock, x, y) {
    //   let maxID = Math.max(0, ...this.blocks.map(o => o.id));
    //   const block = cloneBlock(oldBlock, maxID + 1);
    //   if (!block) {
    //     console.warn('block not create: ' + block);
    //     return;
    //   }
    //   if (x === undefined || y === undefined) {
    //     x = (this.$el.clientWidth / 2 - this.centerX) / this.scale;
    //     y = (this.$el.clientHeight / 2 - this.centerY) / this.scale;
    //   } else {
    //     x = (x - this.centerX) / this.scale;
    //     y = (y - this.centerY) / this.scale;
    //   }
    //   block.position = [x, y];
    //   this.blocks.push(block);
    //   this.blocks = [...this.blocks];
    // },

    // addNewBlock(nodeName, x, y) {
    //   let maxID = Math.max(
    //     0,
    //     ...this.blocks.map(function (o) {
    //       return o.id;
    //     })
    //   );
    //   let block = createBlock(nodeName, maxID + 1);
    //   if (!block) {
    //     console.warn('block not create: ' + block);
    //     return;
    //   }

    //   if (x === undefined || y === undefined) {
    //     x = (this.$el.clientWidth / 2 - this.centerX) / this.scale;
    //     y = (this.$el.clientHeight / 2 - this.centerY) / this.scale;
    //   } else {
    //     x = (x - this.centerX) / this.scale;
    //     y = (y - this.centerY) / this.scale;
    //   }
    //   block.position = [x, y];
    //   this.blocks.push(block);
    //   this.blocks = [...this.blocks];

    //   // this.updateScene();
    // },
    position(block, event) {
      // console.log(block, event)
      block.position = event;
    },
    // deselectAll(withoutID = null) {
    //   this.blocks.forEach(value => {
    //     if (value.id !== withoutID && value.selected) {
    //       this.blockDeselect(value);
    //     }
    //   });
    // },
    // Events
    blockSelect(block) {
      this.$store.dispatch('cascades/deselectBlocks', block);
      this.$nextTick(() => {
        this.$store.dispatch('cascades/selectBlock', block);
      });

      // block.selected = true;
      // this.selectedBlock = block;
      // this.deselectAll(block.id);
      // this.$emit("nodeClick", block.id);
      // this.$emit('blockSelect', block);
    },
    blockDeselect(block) {
      block.selected = false;
      if (block && this.selectedBlock && this.selectedBlock.id === block.id) {
        this.selectedBlock = null;
      }
      this.$emit('blockDeselect', block);
    },
    // blockDelete(block) {
    //   if (block.selected) {
    //     this.blockDeselect(block);
    //   }
    //   this.links.forEach(l => {
    //     if (l.originID === block.id || l.targetID === block.id) {
    //       this.removeLink(l.id);
    //     }
    //   });
    //   this.blocks = this.blocks.filter(b => {
    //     return b.id !== block.id;
    //   });
    //   // this.updateScene();
    // },
    moveBlock() {
      this.updateModel();
    },

    updateModel() {
      this.$store.dispatch('cascades/updateModel');
    },
  },

  mounted() {
    // Context menu off
    // this.$el.addEventListener('contextmenu', event => event.preventDefault());
    this.$el.addEventListener('mouseenter', this.handleMauseOver);
    this.$el.addEventListener('mouseleave', this.handleMauseOver);
    document.documentElement.addEventListener('keyup', this.keyup);
    document.documentElement.addEventListener('mousemove', this.handleMove, true);
    document.documentElement.addEventListener('mousedown', this.handleDown, true);
    document.documentElement.addEventListener('mouseup', this.handleUp, true);
    document.documentElement.addEventListener('wheel', this.handleWheel, true);

    this.centerX = this.$el.clientWidth / 2;
    this.centerY = this.$el.clientHeight / 2;

    // this.importScene();
  },
  beforeDestroy() {
    // this.$el.removeEventListener('contextmenu', null);
    document.documentElement.removeEventListener('keyup', this.keyup);
    this.$el.removeEventListener('mouseenter', this.handleMauseOver);
    this.$el.removeEventListener('mouseleave', this.handleMauseOver);
    document.documentElement.removeEventListener('mousemove', this.handleMove, true);
    document.documentElement.removeEventListener('mousedown', this.handleDown, true);
    document.documentElement.removeEventListener('mouseup', this.handleUp, true);
    document.documentElement.removeEventListener('wheel', this.handleWheel, true);
  },
};
</script>

<style lang="scss" scoped>
.t-block {
  flex-shrink: 1;
  width: 100%;
  background-color: #17212b;
  position: relative;
  overflow: hidden;
  box-sizing: border-box;

  &__lines {
    position: absolute;
  }
  &__center {
    position: absolute;
  }
}
.btn-zoom {
  display: flex;
  flex-direction: column;
  position: absolute;
  bottom: 10px;
  right: 10px;
  background-color: #0e1621;
  border: solid 2px #65b9f4;
  &__item {
    width: 18px;
    height: 18px;
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    i {
      width: 14px;
      height: 14px;
    }
  }
  hr {
    margin: 0;
    border: none;
    color: #65b9f4;
    background-color: #65b9f4;
    height: 2px;
  }
}
</style>
