<template>
  <div class="block-right">
    <div class="block-right__fab">
      <Fab @click="addCard" />
    </div>
    <div class="block-right__header">Выходные параметры</div>
    <div class="block-right__body">
      <scrollbar :ops="ops" ref="scrollRight">
        <div class="block-right__body--inner" :style="height">
          <div class="block-right__body--empty"></div>
          <template v-for="inputData of inputDataOutput">
            <CardLayer
              v-bind="inputData"
              :key="'cardLayersRight' + inputData.id"
              @click-btn="optionsCard($event, inputData.id)"
            >
              <template v-slot:header>Выходные данные {{ inputData.id }}</template>
              <template v-slot:default="{ data: { parameters, errors } }">
                <template v-for="(data, index) of output">
                  <t-auto-field
                    v-bind="data"
                    :parameters="parameters"
                    :errors="errors"
                    :key="inputData.color + index"
                    :idKey="'key_' + index"
                    :id="inputData.id"
                    :update="mixinUpdateDate"
                    root
                    @multiselect="mixinUpdate"
                    @change="mixinChange"
                  />
                </template>
              </template>
            </CardLayer>
          </template>
          <div class="block-right__body--empty"></div>
        </div>
      </scrollbar>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import Fab from '../components/forms/Fab.vue';
import CardLayer from '../components/card/CardLayer.vue';
import blockMain from '@/mixins/datasets/blockMain';
export default {
  name: 'BlockMainRight',
  components: {
    Fab,
    CardLayer,
  },
  mixins: [blockMain],
  data: () => ({
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
      inputData: 'datasets/getInputData',
    }),
    inputDataOutput() {
      return this.inputData.filter(item => {
        return item.layer === 'output';
      });
    },
    height() {
      const height = this.$store.getters['settings/height']({
        clean: true,
        padding: 172 + 90 + 62,
      });
      // console.log(height);
      return height;
    },
  },
  methods: {
    autoScroll() {
      this.$nextTick(() => {
        this.$refs.scrollRight.scrollTo(
          {
            x: '0%',
          },
          100
        );
      });
    },
    addCard() {
      this.$store.dispatch('datasets/createInputData', { layer: 'output' });
      this.autoScroll();
    },
    optionsCard(comm, id) {
      if (comm === 'remove') {
        this.$store.dispatch('datasets/removeInputData', id);
        this.mixinRemove(id);
      }
      if (comm === 'copy') {
        this.$store.dispatch('datasets/cloneInputData', id);
        this.autoScroll();
      }
    },
  },
  mounted() {
    // console.log(this.output);
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
    height: 32px;
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
    padding: 4px 40px;
    user-select: none;
  }
  &__body {
    width: 100%;
    padding: 40px 0px 0px 0px;
    overflow: auto;
    &--inner {
      display: flex;
      width: 100%;
      justify-content: flex-end;
      overflow: auto;
      height: 100%;
      flex-direction: row-reverse;
    }
    &--empty {
      height: 100%;
      width: 3px;
    }
  }
  &__fab {
    position: absolute;
    left: 6px;
    top: 4px;
    z-index: 100;
  }
}
</style>
