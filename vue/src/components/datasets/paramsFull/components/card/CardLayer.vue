<template>
  <div class="card-layer" :style="height">
    <div class="card-layer__header" :style="bg" @click="$emit('click-header', $event)">
      <div class="card-layer__header--icon" v-click-outside="outside" @click="toggle = !toggle">
        <i class="t-icon icon-file-dot"></i>
      </div>
      <div class="card-layer__header--title"><slot name="header" :id="id"></slot></div>
    </div>
    <div v-show="toggle" class="card-layer__dropdown">
      <div v-for="({ icon }, i) of items" :key="'icon' + i" class="card-layer__dropdown--item" @click="click(icon)">
        <i :class="[icon]"></i>
      </div>
    </div>
    <div class="card-layer__body">
      <scrollbar :ops="ops">
        <div class="card-layer__body--inner" ref="cardBody">
          <slot :data="data" />
        </div>
      </scrollbar>
    </div>
  </div>
</template>

<script>
export default {
  name: 'card-layer',
  props: {
    id: Number,
    layer: String,
    name: String,
    type: String,
    color: String,
    parameters: {
      type: Object,
      default: () => {},
    },
  },
  data: () => ({
    height: { height: '100%' },
    toggle: false,
    items: [{ icon: 'remove' }, { icon: 'copy' }],
    ops: {
      bar: { background: '#17212b' },
      scrollPanel: {
        scrollingX: false,
        scrollingY: true,
      },
    },
  }),
  computed: {
    errors() {
      return this.$store.getters['datasets/getErrors'](this.id);
    },
    data() {
      return { errors: this.errors, parameters: { ...this.parameters, name: this.name, type: this.type } };
    },
    bg() {
      return { backgroundColor: this.color };
    },
  },
  methods: {
    outside() {
      if (this.toggle) {
        this.toggle = false;
      }
    },
    click(icon) {
      this.toggle = false;
      this.$emit('click-btn', icon);
    },
  },
  mounted() {
    const heightCard = this.$el.clientHeight;
    const heightBody = this.$refs.cardBody.clientHeight + 36;
    if (heightCard > heightBody) {
      this.height = { height: heightBody + 'px' };
    }
    this.$emit('mount', true);
  },
};
</script>

<style lang="scss" scoped>
.card-layer {
  flex: 0 0 242px;
  width: 242px;
  position: relative;
  border-radius: 4px;
  border: 1px solid #6c7883;
  background-color: #242f3d;
  margin: 0 3px;
  min-height: 100%;
  &__body {
    width: 100%;
    padding-top: 34px;
    // padding: 30px 8px 16px 8px;
    position: relative;
    height: 100%;
    overflow: hidden;
    &--inner {
      // position: absolute;
      // height: 100%;
      padding: 0px 8px 8px 8px;
    }
  }
  &__header {
    user-select: none;
    position: absolute;
    top: 0;
    height: 24px;
    background-color: #6c7883;
    width: 100%;
    border-radius: 3px 3px 0 0;
    font-family: Open Sans;
    font-style: normal;
    font-weight: normal;
    font-size: 12px;
    line-height: 16px;
    display: flex;
    align-items: center;
    padding: 2px 6px 4px 6px;
    z-index: 1;
    &--icon {
      cursor: pointer;
      position: absolute;
      right: 12px;
      height: 100%;
      display: flex;
      align-items: center;
      & .t-icon {
        width: 16px;
        height: 6px;
      }
    }
  }
  &__dropdown {
    position: absolute;
    background-color: #2b5278;
    border-radius: 4px;
    right: 3px;
    top: 3px;
    z-index: 100;
    &--item {
      position: relative;
      width: 32px;
      height: 32px;
      display: flex;
      justify-content: center;
      align-items: center;
      cursor: pointer;
      &:hover {
        opacity: 0.7;
      }
      & .remove {
        display: inline-block;
        width: 14px;
        height: 18px;
        background-repeat: no-repeat;
        background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTgiIHZpZXdCb3g9IjAgMCAxNCAxOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEgMTZDMSAxNy4xIDEuOSAxOCAzIDE4SDExQzEyLjEgMTggMTMgMTcuMSAxMyAxNlY2QzEzIDQuOSAxMi4xIDQgMTEgNEgzQzEuOSA0IDEgNC45IDEgNlYxNlpNNCA2SDEwQzEwLjU1IDYgMTEgNi40NSAxMSA3VjE1QzExIDE1LjU1IDEwLjU1IDE2IDEwIDE2SDRDMy40NSAxNiAzIDE1LjU1IDMgMTVWN0MzIDYuNDUgMy40NSA2IDQgNlpNMTAuNSAxTDkuNzkgMC4yOUM5LjYxIDAuMTEgOS4zNSAwIDkuMDkgMEg0LjkxQzQuNjUgMCA0LjM5IDAuMTEgNC4yMSAwLjI5TDMuNSAxSDFDMC40NSAxIDAgMS40NSAwIDJDMCAyLjU1IDAuNDUgMyAxIDNIMTNDMTMuNTUgMyAxNCAyLjU1IDE0IDJDMTQgMS40NSAxMy41NSAxIDEzIDFIMTAuNVoiIGZpbGw9IiNBN0JFRDMiLz4KPC9zdmc+Cg==');
      }
      & .copy {
        display: inline-block;
        width: 17px;
        height: 20px;
        background-repeat: no-repeat;
        background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTciIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAxNyAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTExLjYzMTYgMEgxLjc4OTQ3QzAuODA1MjYzIDAgMCAwLjgxODE4MiAwIDEuODE4MThWMTMuNjM2NEMwIDE0LjEzNjQgMC40MDI2MzIgMTQuNTQ1NSAwLjg5NDczNyAxNC41NDU1QzEuMzg2ODQgMTQuNTQ1NSAxLjc4OTQ3IDE0LjEzNjQgMS43ODk0NyAxMy42MzY0VjIuNzI3MjdDMS43ODk0NyAyLjIyNzI3IDIuMTkyMTEgMS44MTgxOCAyLjY4NDIxIDEuODE4MThIMTEuNjMxNkMxMi4xMjM3IDEuODE4MTggMTIuNTI2MyAxLjQwOTA5IDEyLjUyNjMgMC45MDkwOTFDMTIuNTI2MyAwLjQwOTA5MSAxMi4xMjM3IDAgMTEuNjMxNiAwWk0xNS4yMTA1IDMuNjM2MzZINS4zNjg0MkM0LjM4NDIxIDMuNjM2MzYgMy41Nzg5NSA0LjQ1NDU1IDMuNTc4OTUgNS40NTQ1NVYxOC4xODE4QzMuNTc4OTUgMTkuMTgxOCA0LjM4NDIxIDIwIDUuMzY4NDIgMjBIMTUuMjEwNUMxNi4xOTQ3IDIwIDE3IDE5LjE4MTggMTcgMTguMTgxOFY1LjQ1NDU1QzE3IDQuNDU0NTUgMTYuMTk0NyAzLjYzNjM2IDE1LjIxMDUgMy42MzYzNlpNMTQuMzE1OCAxOC4xODE4SDYuMjYzMTZDNS43NzEwNSAxOC4xODE4IDUuMzY4NDIgMTcuNzcyNyA1LjM2ODQyIDE3LjI3MjdWNi4zNjM2NEM1LjM2ODQyIDUuODYzNjQgNS43NzEwNSA1LjQ1NDU1IDYuMjYzMTYgNS40NTQ1NUgxNC4zMTU4QzE0LjgwNzkgNS40NTQ1NSAxNS4yMTA1IDUuODYzNjQgMTUuMjEwNSA2LjM2MzY0VjE3LjI3MjdDMTUuMjEwNSAxNy43NzI3IDE0LjgwNzkgMTguMTgxOCAxNC4zMTU4IDE4LjE4MThaIiBmaWxsPSIjQTdCRUQzIi8+Cjwvc3ZnPgo=');
      }
    }
  }
}
</style>
