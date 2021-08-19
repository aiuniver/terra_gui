<template>
  <div class="block-right">
    <div class="block-right__fab">
      <Fab @click="addCard" />
    </div>
    <div class="block-right__header">Выходные параметры</div>
    <div class="block-right__body">
      <scrollbar :ops="ops" ref="scrollRight">
        <div class="block-right__body--inner" :style="height">
          <template v-for="{ id, color } of inputDataOutput">
            <CardLayer :id="id" :color="color" :key="'cardLayersRight' + id" @click-btn="optionsCard($event, id)">
              <template v-slot:header>Выходные данные {{ id }}</template>
              <TMultiSelect
                :id="id"
                :lists="mixinFiles"
                label="Выберите путь"
                inline
                @change="mixinCheck($event, id)"
              />
              <template v-for="(data, index) of output">
                <t-auto-field v-bind="data" :key="color + index" :idKey="color + index" :id="id" root @change="mixinChange" />
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
import TMultiSelect from '../../../forms/MultiSelect.vue';
import blockMain from '@/mixins/datasets/blockMain';
export default {
  name: 'BlockMainRight',
  components: {
    Fab,
    CardLayer,
    TMultiSelect,
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
    addCard() {
      this.$store.dispatch('datasets/createInputData', { layer: 'output' });
      this.$nextTick(() => {
        this.$refs.scrollRight.scrollTo(
          {
            x: '0%',
          },
          100
        );
      });
    },
    optionsCard(comm, id) {
      if (comm === 'remove') {
        this.$store.dispatch('datasets/removeInputData', id);
        this.mixinRemove(id);
      }
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