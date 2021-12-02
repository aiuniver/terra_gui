<template>
  <div class="card">
    <div class="card__content">
      <div v-if="type == 'ImageClassification'">
        <div class="card__original">
          <ImgCard :imgUrl="card.source" />
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '224px', height: '80px' }">{{ classificationResult }}</TextCard>
        </div>
      </div>
      <div v-if="type == 'TextClassification'">
        <div class="card__original">
          <TextCard :style="{ width: '600px', color: '#A7BED3', height: '324px' }">{{ card.source }}</TextCard>
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '600px', height: '80px' }">{{ classificationResult }}</TextCard>
        </div>
      </div>
      <div v-if="type == 'TextSegmentation'">
        <div class="card__original segmentation__original" :style="{ height: '324px' }">
          <scrollbar :ops="ops">
            <TableTextSegmented
              v-bind="{
                value: card.format,
                tags_color: { segmentationLayer },
                layer: 'segmentationLayer',
                block_width: '598px',
              }"
            />
          </scrollbar>
        </div>
        <div class="card__result">
          <SegmentationTags :style="{ width: '600px', height: '50px' }" :tags="segmentationLayer" />
        </div>
      </div>
      <div v-if="type == 'AudioClassification'">
        <div class="card__original">
          <AudioCard :value="card.source" />
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '600px', height: '80px' }">{{ classificationResult }}</TextCard>
        </div>
      </div>

      <div v-if="type == 'ImageSegmentation'">
        <div class="card__original">
          <ImgCard :imgUrl="card.source" />
        </div>
        <div class="card__result">
          <ImgCard :imgUrl="card.segment" />
        </div>
      </div>
      <div v-if="type == 'VideoObjectDetection'">
        <div class="card__original">
          <TableVideo :value="card.source" />
        </div>
        <div class="card__result">
          <TableVideo :value="card.predict" />
        </div>
      </div>
      <div v-if="type == 'YoloV3' || type == 'YoloV4'">
        <div class="card__original">
          <TableImage size="large" :value="card.source" />
        </div>
        <div class="card__result">
          <TableImage size="large" :value="card.predict" />
        </div>
      </div>
      <div class="card__graphic" v-if="type == 'Timeseries'">
        <GraphicCard v-bind="card" :key="'graphic_' + index" />
      </div>
      <div class="card__graphic" v-if="type == 'TimeseriesTrend'">
        <div class="card__original">
          <GraphicCardPredict :data="card.predict" />
        </div>
        <div class="card__result">
          <GraphicCardSource :data="card.source" />
        </div>
      </div>
    </div>
    <div class="card__reload">
      <button class="btn-reload" @click="reload">
        <i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i>
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'IndexCard',
  components: {
    ImgCard: () => import('@/components/deploy/cards/ImgCard'),
    TableVideo: () => import('@/components/deploy/cards/TableVideo'),
    TextCard: () => import('@/components/deploy/cards/TextCard'),
    GraphicCard: () => import('@/components/deploy/cards/GraphicCard'),
    GraphicCardSource: () => import('@/components/deploy/cards/GraphicCardSource'),
    GraphicCardPredict: () => import('@/components/deploy/cards/GraphicCardPredict'),
    AudioCard: () => import('@/components/deploy/cards/AudioCard'),
    TableTextSegmented: () => import('../training/main/prediction/components/TableTextSegmented'),
    SegmentationTags: () => import('@/components/deploy/cards/SegmentationTags'),
    TableImage: () => import('@/components/training/main/prediction/components/TableImage'),
  },
  data: () => ({
    ops: {
      scrollPanel: {
        scrollingX: false,
        scrollingY: true,
      },
    },
  }),
  props: {
    card: {
      type: Object,
      default: () => ({}),
    },
    colorMap: {
      type: Array,
      default: () => [],
    },
    index: {
      type: [String, Number],
      required: true
    },
    defaultLayout:{
      type: Object,
      default: () => ({})
    },
    type: {
      type: String,
      default: ""
    },
  },

  methods: {
    async reload() {
      this.$emit('reload', String(this.index));
    },
  },
  computed: {
    layout() {
      const layout = this.defaultLayout;
      if (this.char) {
        layout.title.text = this.char.title || '';
        layout.xaxis.title = this.char.xaxis.title || '';
        layout.yaxis.title = this.char.yaxis.title || '';
      }
      return layout;
    },
    segmentationLayer() {
      const layer = {};
      for (let i in this.colorMap) {
        if (this.colorMap[i][0].includes('p')) continue;
        const tag = this.colorMap[i][0].slice(1, this.colorMap[i][0].length - 1);
        layer[tag] = this.colorMap[i][2];
      }
      return layer;
    },
    classificationResult() {
      let text = this.card.data
      let prepareText = ''
      text.sort((a, b) => (a[1] < b[1] ? 1 : -1));
      for (let i = 0; i < text.length; i++) prepareText += `${text[i][0]} - ${text[i][1]}% \n`;
      return prepareText;
    },
  },
};
</script>

<style lang="scss" scoped>
.card__reload {
  padding-left: 5px;
}
.card {
  padding: 15px 15px 15px 0;
  display: flex;
}
.card__graphic {
  background: #242f3d;
  border: 1px solid #6c7883;
  box-sizing: border-box;
  border-radius: 4px;
}
.card__original {
  background: #242f3d;
}
.card__result {
  padding-top: 6px;
}
.btn-reload {
  width: 32px;
  height: 32px;
  i {
    position: absolute;
    margin-left: 7px;
    margin-top: -13px;
    width: 16px;
  }
}
.card__table {
  width: 100%;
}
.segmentation {
  &__original {
    border: 1px solid #6c7883;
    border-radius: 4px;
  }
}
</style>