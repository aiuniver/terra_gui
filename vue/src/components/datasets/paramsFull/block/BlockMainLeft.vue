<template>
  <div class="block-left">
    <div class="block-left__fab">
      <Fab @click="add" />
    </div>
    <div class="block-left__header">Входные параметры</div>
    <div class="block-left__body">
      <scrollbar :ops="ops" ref="scrollLeft">
        <div class="block-left__body--inner" :style="height">
          <template v-for="({ title, color }, i) of cardLayers">
            <CardLayer
              :title="title + ' ' + (i + 1)"
              :color="color"
              :key="'cardLayersLeft' + i"
              @click-btn="click($event, i)"
            >
              <t-select label="Выберите путь" :lists="filesDrop" name="path" @change="change" />
              <template v-for="(data, index) of input">
                <t-auto-field v-bind="data" @change="change" :key="color + index" :idKey="color + index" />
              </template>
            </CardLayer>
          </template>
          <div class="block-left__body--empty"></div>
        </div>
      </scrollbar>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import { getColor } from '../util/color';
import Fab from '../components/forms/Fab.vue';
import CardLayer from '../components/card/CardLayer.vue';
export default {
  name: 'BlockMainLeft',
  components: {
    Fab,
    CardLayer,
  },
  data: () => ({
    cardLayers: [{ title: 'Input', color: '#FFB054' }],
    ops: {
      scrollPanel: {
        scrollingX: true,
        scrollingY: false,
      },
      rail: {
        gutterOfEnds: '6px',
      },
    },
  }),
  computed: {
    ...mapGetters({
      input: 'datasets/getTypeInput',
    }),
    filesDrop: {
      set(value) {
        this.$store.dispatch('datasets/setFilesDrop', value);
      },
      get() {
        const files = this.$store.getters['datasets/getFilesDrop'];
        return files.map(({ title, path}) => {
          return { label: title, value: path}
        })
      },
    },
    height() {
      const height = this.$store.getters['settings/height']({
        clean: true,
        padding: 172 + 90 + 62,
      });
      console.log(height);
      return height;
    },
  },
  methods: {
    add() {
      this.cardLayers.push({ title: 'Input', color: getColor() });
      this.$nextTick(() => {
        this.$refs.scrollLeft.scrollTo(
          {
            x: '100%',
          },
          100
        );
      });
    },
    click(comm, index) {
      console.log(comm, index);
      if (comm === 'remove') {
        this.cardLayers = this.cardLayers.filter((_, i) => {
          return i !== index;
        });
      }
    },
    change(e) {
      console.log(e);
    },
    heightForm(value) {
      // console.log(value, this.$el.clientHeight);
      // const clearHeight = this.$el.clientHeight - 56;
      this.$store.dispatch('settings/setHeight', {
        center: this.$el.clientHeight,
      });
      console.log(value, this.$el.clientHeight);
      // this.height = value > clearHeight ? clearHeight : value + 56;
      // this.height = clearHeight
    },
  },
};
</script>

<style lang="scss" scoped>
.block-left {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
  height: 100%;
  &__header {
    position: absolute;
    height: 24px;
    width: 100%;
    top: 0;
    background: #242f3d;
    font-family: Open Sans;
    font-style: normal;
    font-weight: normal;
    font-size: 12px;
    line-height: 16px;
    display: flex;
    align-items: center;
    text-align: center;
    color: #ffffff;
    padding: 4px 16px;
    justify-content: flex-end;
  }
  &__body {
    width: 100%;
    /* position: absolute; */
    /* height: 250px; */
    /* top: 34px; */
    padding: 40px 0px 0px 0px;
    /* right: 70px; */
    overflow: auto;
    &--inner {
      display: flex;
      width: 100%;
      justify-content: flex-end;
      overflow: auto;
      // position: absolute;
      height: 100%;
    }
    &--empty {
      height: 100%;
      width: 70px;
    }
  }
  &__fab {
    position: absolute;
    right: 16px;
    top: 40px;
    z-index: 100;
  }
}
</style>