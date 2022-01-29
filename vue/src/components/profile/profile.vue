<template>
        <DModal v-model="DialogProfile" title="Мой профиль">
      <t-field class="profile-modal-content__wrapper" label="Имя *">
        <DInputText v-model.trim="firstName" :error="errFirst" @input="errFirst = ''" />
      </t-field>
      <t-field class="profile-modal-content__wrapper" label="Фамилия *">
        <DInputText v-model.trim="lastName" :error="errLast" @input="errLast = ''" />
      </t-field>
      <t-field class="profile-modal-content__wrapper" label="Логин">
        <p>{{ user.login }}</p>
      </t-field>
      <t-field class="profile-modal-content__wrapper" label="E-mail">
        <p>{{ user.email }}</p>
      </t-field>
      <t-field class="profile-modal-content__wrapper" label="Token">
        <p>
          <span ref="token">{{ user.token }}</span>
          <svg
            @click="copy"
            class="btn-copy"
            width="21"
            height="21"
            viewBox="0 0 21 21"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M12.9999 20.9999H2.99995C2.46402 21.0186 1.94442 20.8138 1.56524 20.4347C1.18605 20.0555 0.981324 19.5359 0.999948 18.9999V8.99995C0.981324 8.46402 1.18605 7.94442 1.56524 7.56524C1.94442 7.18605 2.46402 6.98132 2.99995 6.99995H6.99995V2.99995C6.98132 2.46402 7.18605 1.94442 7.56524 1.56524C7.94442 1.18605 8.46402 0.981324 8.99995 0.999948H18.9999C19.5359 0.981324 20.0555 1.18605 20.4347 1.56524C20.8138 1.94442 21.0186 2.46402 20.9999 2.99995V12.9999C21.0183 13.5358 20.8134 14.0552 20.4343 14.4343C20.0552 14.8134 19.5358 15.0183 18.9999 14.9999H14.9999V18.9999C15.0183 19.5358 14.8134 20.0552 14.4343 20.4343C14.0552 20.8134 13.5358 21.0183 12.9999 20.9999ZM2.99995 8.99995V18.9999H12.9999V14.9999H8.99995C8.46411 15.0183 7.94468 14.8134 7.56557 14.4343C7.18645 14.0552 6.98162 13.5358 6.99995 12.9999V8.99995H2.99995ZM8.99995 2.99995V12.9999H18.9999V2.99995H8.99995Z"
            />
          </svg>
        </p>
        <p class="copy-msg"></p>
        <div @click="updateTokenProfile" class="btn-text">Обновить токен</div>
      </t-field>
      <t-field class="profile-modal-content__wrapper" v-show="showNoticeProfile">
        <div>
          <i class="notice__icon"></i>
          <p>{{ noticeMsgProfile }}</p>
        </div>
      </t-field>
      <template slot="footer">
        <DButton
          style="margin-left: auto"
          @click="saveProfile"
          :loading="isLoadingProfile"
          :disabled="isLoadingProfile"
          color="primary"
          direction="left"
        >
          Сохранить
        </DButton>
      </template>
    </DModal>
</template>