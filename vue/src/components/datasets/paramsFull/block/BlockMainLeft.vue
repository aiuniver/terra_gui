<template>
  <div class="block-left">
    <div class="block-left__fab">
      <Fab @click="addCard" />
    </div>
    <div class="block-left__header">Входные параметры</div>
    <div class="block-left__body">
      <scrollbar :ops="ops" ref="scrollLeft">
        <div class="block-left__body--inner" :style="height">
          <template v-for="{ id, color } of inputDataInput">
            <CardLayer
              :id="id"
              :color="color"
              :key="'cardLayersLeft' + id"
              @click-btn="optionsCard($event, id)"
            >
              <template v-slot:header>Входные данные {{ id }}</template>
              <TMultiSelect
                :id="id"
                :lists="mixinFiles"
                label="Выберите путь"
                inline
                @change="mixinCheck($event, id)"
              />
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
import Fab from '../components/forms/Fab.vue';
import CardLayer from '../components/card/CardLayer.vue';
import TMultiSelect from '@/components/forms/MultiSelect.vue';
import blockMain from '@/mixins/datasets/blockMain';

export default {
  name: 'BlockMainLeft',
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
      input: 'datasets/getTypeInput',
      inputData: 'datasets/getInputData',
    }),
    inputDataInput() {
      return this.inputData.filter(item => {
        return item.layer === 'input';
      });
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
    addCard() {
      this.$store.dispatch('datasets/createInputData', { layer: 'input' });
      this.$nextTick(() => {
        this.$refs.scrollLeft.scrollTo(
          {
            x: '100%',
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
    change(e) {
      console.log(e);
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