<template>
  <div class="block-left">
    <div class="block-left__fab">
      <Fab @click="addCard" />
    </div>
    <div class="block-left__header">Входные параметры</div>
    <div class="block-left__body">
      <scrollbar :ops="ops" ref="scrollLeft">
        <div class="block-left__body--inner" :style="height">
          <div class="block-left__body--empty"></div>
          <template v-for="inputData of inputDataInput">
            <CardLayer
              v-bind="inputData"
              :key="'cardLayersLeft' + inputData.id"
              @click-btn="optionsCard($event, inputData.id)"
            >
              <template v-slot:header>Входные данные {{ inputData.id }}</template>
              <template v-slot:default="{ data: { parameters, errors } }">
                <template v-for="(data, index) of input">
                  <t-auto-field
                    v-bind="data"
                    :parameters="parameters"
                    :errors="errors"
                    :key="inputData.color + index"
                    :idKey="'key_' + index"
                    :id="inputData.id"
                    :update="mixinUpdateDate"
                    :isAudio="isAudio"
                    root
                    @multiselect="mixinUpdate"
                    @change="mixinChange"
                  />
                </template>
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
import blockMain from '@/mixins/datasets/blockMain';
// import Error from '@/utils/core/Errors'

export default {
  name: 'BlockMainLeft',
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
      input: 'datasets/getTypeInput',
      inputData: 'datasets/getInputData',
    }),
    isAudio() {
      const [audio] = this.inputDataInput.filter(item => item.type === 'Audio');
      return audio?.id;
    },
    inputDataInput() {
      const arr = this.inputData.filter(item => {
        return item.layer === 'input';
      });
      return arr;
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
    error(id, key) {
      const errors = this.$store.getters['datasets/getErrors'](id);
      return errors?.[key]?.[0] || errors?.parameters?.[key]?.[0] || '';
    },
    autoScroll() {
      this.$nextTick(() => {
        this.$refs.scrollLeft.scrollTo(
          {
            x: '100%',
          },
          100
        );
      });
    },
    addCard() {
      this.$store.dispatch('datasets/createInputData', { layer: 'input' });
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
    justify-content: flex-end;
    user-select: none;
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
      width: 3px;
    }
  }
  &__fab {
    position: absolute;
    right: 6px;
    top: 4px;
    z-index: 100;
  }
}
</style>
