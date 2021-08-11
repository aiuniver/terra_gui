<template>
  <div class="block-right">
    <div class="block-right__fab">
      <Fab @click="add" />
    </div>
    <div class="block-right__header">Выходные параметры</div>
    <div class="block-right__body">
      <scrollbar :ops="ops" ref="scrollRight">
        <div class="block-right__body--inner" :style="height">
          <div class="block-right__body--empty"></div>
          <template v-for="({ title, color }, i) of cardLayers">
            <CardLayer
              :title="title + ' ' + (i + 1)"
              :color="color"
              :key="'cardLayersRight' + i"
              @click-btn="click($event, i)"
              @click-header="clickScroll"
            >
              <t-select label="Выберите путь" :lists="filesDrop" name="path" @change="change" />
              <template v-for="(data, index) of output">
                <t-auto-field v-bind="data" @change="change" :key="color + index" :idKey="color + index" />
              </template>
            </CardLayer>
          </template>
        </div>
      </scrollbar>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
import { getColor } from '../util/color';
import Fab from '../components/forms/Fab.vue';
import CardLayer from '../components/card/CardLayer.vue';
export default {
  name: 'BlockMainRight',
  components: {
    Fab,
    CardLayer,
  },
  data: () => ({
    cardLayers: [{ title: 'Output', color: '#8e51f2' }],
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
      output: 'datasets/getTypeOutput',
    }),
    filesDrop: {
      set(value) {
        this.$store.dispatch('datasets/setFilesDrop', value);
      },
      get() {
        return this.$store.getters['datasets/getFilesDrop'];
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
      this.cardLayers.push({ title: 'Output', color: getColor() });
      this.$nextTick(() => {
        this.$refs.scrollRight.scrollTo(
          {
            x: "100%",
          },
          100
        );
      });
    },
    clickScroll(e) {
      this.$refs.scrollRight.scrollIntoView(e.target, 100);
      console.log(e);
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
  },
};
</script>

<style lang="scss" scoped>
.block-right {
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
      justify-content: flex-start;
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
    left: 16px;
    top: 40px;
    z-index: 100;
  }
}
</style>