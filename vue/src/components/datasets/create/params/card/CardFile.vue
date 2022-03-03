<template>
  <div class="t-card-file" :style="bc" v-click-outside="outside">
    <div v-if="id" class="t-card-file__header" :style="bg">{{ title }}</div>
    <div :class="['t-card-file__body', 'icon-file-' + type]" :style="img"></div>
    <div class="t-card-file__footer">
      <div class="t-card-file__footer--label">{{ label }}</div>
      <!-- <div class="t-card-file__footer--btn" @click="show = true">
        <i class="t-icon icon-file-dot"></i>
      </div> -->
    </div>
    <!-- <div v-show="show" class="t-card-file__dropdown">
      <div
        v-for="({ icon, event }, i) of items"
        :key="'icon' + i"
        class="t-card-file__dropdown--item"
        @click="$emit('event', { label, event }), (show = false)"
      >
        <i :class="['t-icon', icon]"></i>
      </div>
    </div> -->
  </div>
</template>

<script>
export default {
  name: 't-card-file',
  props: {
    label: String,
    type: String,
    id: Number,
    cover: String,
  },
  data: () => ({
    show: false,
    items: [{ icon: 'icon-deploy-remove', event: 'remove' }],
  }),
  computed: {
    img() {
      return this.cover
        ? { backgroundImage: `url('${this.cover}')`, backgroundPosition: 'center', backgroundSize: 'cover' }
        : {};
    },
    selectInputData() {
      return this.$store.getters['datasets/getInputDataByID'](this.id) || {};
    },
    title() {
      const card = this.selectInputData;
      return card.layer === 'input' ? 'Входные данные ' + card.id : 'Выходные данные ' + card.id;
    },
    color() {
      return this.selectInputData.color || '';
    },
    bg() {
      return { backgroundColor: this.id ? this.color : '' };
    },
    bc() {
      return { borderColor: this.id ? this.color : '' };
    },
  },
  methods: {
    outside() {
      this.show = false;
    },
  },
};
</script>

<style lang="scss" scoped>
.t-card-file {
  flex: 0 0 152px;
  position: relative;
  width: 152px;
  height: 152px;
  border: 1px solid #6c7883;
  box-sizing: border-box;
  border-radius: 4px;
  background-color: #17212b;
  margin: 0 5px;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  overflow: hidden;
  &__footer,
  &__header {
    position: relative;
    height: 24px;
    background-color: #242f3d;
    width: 100%;
    border-radius: 0 0 3px 3px;
    font-family: Open Sans;
    font-style: normal;
    font-weight: normal;
    font-size: 12px;
    line-height: 16px;
  }
  &__header {
    position: absolute;
    top: 0;
    border-radius: 3px 3px 0 0;
    padding: 2px 6px 4px 6px;
  }
  &__body {
    height: 100%;
    width: 100%;
    display: inline-block;
    background-repeat: no-repeat;
    background-position: 50% 40%;
    background-size: 39px 39px;
  }
  &__footer {
    display: flex;
    justify-content: space-between;

    &--label {
      bottom: 0;
      border-radius: 0 0 3px 3px;
      padding: 4px 2px 2px 6px;
      text-overflow: ellipsis;
      overflow: hidden;
      white-space: nowrap;
    }
    &--btn {
      padding: 0 6px 0 0;
      cursor: pointer;
      i {
        width: 16px;
      }
      &:hover {
      }
    }
  }
  &__dropdown {
    position: absolute;
    background-color: #2b5278;
    border-radius: 4px;
    right: 3px;
    bottom: 3px;
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
      i {
        width: 14px;
      }
    }
  }
}
</style>
