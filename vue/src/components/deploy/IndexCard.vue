<template>
  <div class="card">
    <div class="card__content">
      <div v-if="['image_gan', 'image_cgan'].includes(type)">
        <div class="card__original">
          <ImgCard :imgUrl="card.source" />
        </div>
        <div>
          <TextCard :style="{ width: '224px', height: '20px' }">
            {{ card.actual }}
          </TextCard>
        </div>
      </div>
      <div v-if="type == 'image_classification'">
        <div class="card__original">
          <ImgCard :imgUrl="card.source" />
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '224px', height: '80px' }">
            <template v-for="{ name, value } of getData">
              <div class="video_classification__item" :key="name">{{ `${name}: ${value}%` }}</div>
            </template>
          </TextCard>
        </div>
      </div>
      <div v-if="type == 'video_classification'">
        <div class="card__original">
          <TableVideo :value="card.source" />
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '300px', height: '80px' }">
            <div class="video_classification">
              <template v-for="{ name, value } of getData">
                <div class="video_classification__item" :key="name">{{ `${name}: ${value}%` }}</div>
              </template>
            </div>
          </TextCard>
        </div>
      </div>
      <div v-if="type == 'text_classification'">
        <div class="card__original">
          <TextCard :style="{ width: '600px', color: '#A7BED3', height: '324px' }">{{ card.source }}</TextCard>
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '600px', height: '80px' }">
            <div v-for="{ name, value } of getData" :key="name">{{ `${name}: ${value}%` }}</div>
          </TextCard>
        </div>
      </div>
      <div v-if="type == 'text_segmentation'">
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
      <div v-if="type == 'audio_classification'">
        <div class="card__original">
          <AudioCard :value="card.source" :key="card.source" />
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '600px', height: '80px' }">
            <div v-for="{ name, value } of getData" :key="name">{{ `${name}: ${value}%` }}</div>
          </TextCard>
        </div>
      </div>
      <div v-if="type == 'text_to_audio'">
        <div class="card__result">
          <TextCard :style="{ width: '600px', height: '200px' }">
            {{ card.source }}
          </TextCard>
        </div>
        <div class="card__original">
          <AudioCard :value="card.predict" :key="card.source" />
        </div>
      </div>
      <div v-if="type == 'audio_to_text'">
        <div class="card__original">
          <AudioCard :value="card.source" :key="card.predict" />
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '600px', height: '200px' }">
            {{ card.predict }}
          </TextCard>
        </div>
      </div>

      <div v-if="type == 'image_segmentation'">
        <div class="card__original">
          <ImgCard :imgUrl="card.source" />
        </div>
        <div class="card__result">
          <ImgCard :imgUrl="card.segment" />
        </div>
      </div>
      <div v-if="type == 'video_object_detection'">
        <div class="card__original">
          <TableVideo :value="card.source" />
        </div>
        <div class="card__result">
          <TableVideo :value="card.predict" />
        </div>
      </div>
      <div v-if="type == 'object_detection'">
        <div class="card__original">
          <TableImage size="large" :value="card.source" />
        </div>
        <div class="card__result">
          <TableImage size="large" :value="card.predict" />
        </div>
      </div>
      <div class="card__graphic" v-if="type == 'time_series'">
        <GraphicCard v-bind="card" :key="'graphic_' + index" />
      </div>
      <div class="card__graphic" v-if="type == 'time_series_trend'">
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
// ImageSegmentation = "image_segmentation"
// ImageClassification = "image_classification"
// TextSegmentation = "text_segmentation"
// TextClassification = "text_classification"
// AudioClassification = "audio_classification"
// VideoClassification = "video_classification"
// DataframeRegression = "table_data_regression"
// DataframeClassification = "table_data_classification"
// Timeseries = "time_series"
// TimeseriesTrend = "time_series_trend"
// VideoObjectDetection = "video_object_detection"
// YoloV3 = "object_detection"
// YoloV4 = "object_detection"
// YoloV5 = "object_detection"
// GoogleTTS = "text_to_audio"
// Wav2Vec = "audio_to_text"
// TinkoffAPI = "audio_to_text"

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
      required: true,
    },
    defaultLayout: {
      type: Object,
      default: () => ({}),
    },
    type: {
      type: String,
      default: '',
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
    // classificationResult() {
    //   let text = this.card.data;
    //   let prepareText = '';
    //   text.sort((a, b) => (a[1] < b[1] ? 1 : -1));
    //   for (let i = 0; i < text.length; i++) prepareText += `${text[i][0]} - ${text[i][1]}%`;
    //   return prepareText;
    // },
    getData() {
      const arr = this.card?.data || [];
      const text = arr.map(i => ({ name: i[0], value: i[1] }));

      return text;
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

  color: #fff;
  background-color: #2b5278;
  box-shadow: 0 1px 3px 0 rgb(0 133 255 / 50%);

  border: 1px solid #65b9f4;
  border-radius: 4px;
  cursor: pointer;
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
.video_classification {
  position: relative;
  display: flex;
  flex-wrap: wrap;
  &__item {
    flex: 1 1 45%;
  }
}
</style>